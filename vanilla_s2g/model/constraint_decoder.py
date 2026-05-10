"""
Constraint decoder for the S2G model.

Implements the finite-state machine (FSM) described in Section 6.1 of
the specification.  The FSM restricts the decoder's vocabulary at each
generation step to enforce a valid SEL expression.

Three constraint mechanisms are composed:

1. **Label Name Trie** — After ``<rel>``, the decoder can only follow
   valid paths through a prefix tree.  The trie terminates with
   ``<tail>`` as a sentinel.

2. **Source Sequence Copy** — After ``<ent>`` or ``<tail>``, the decoder
   can only generate tokens that appear in the source sentence, forming
   exact substrings of the input.

3. **Null State** — After ``<null>``, the decoder follows a separate
   trie (same labels but terminates with ``<null>`` or ``EOS`` instead
   of ``<tail>``).

Per-instance label tries
------------------------
The label tries are built **per batch item**, directly from the encoder
input (``source_ids``).  Each row of ``source_ids`` carries the SSI
prefix that was actually shown to the encoder, structured as::

    <type> tok* <type> tok* ... <type> tok* <text> raw text tokens

The processor walks each row, slicing out the token-ID spans between
``<type>`` markers (terminating at ``<text>``), and feeds those spans
straight into the trie.  No round-trip through string tokenisation
occurs, so the trie's paths are guaranteed to coincide with the exact
token IDs the model produced for the same labels at decoding time.

A consequence is that the constraint decoder restricts outputs to the
relation types **prompted in that specific instance's SSI**, not to any
externally supplied schema list.  This is the correct semantics for
both evaluation (where the SSI is capped by ``ssi.max_types_in_prompt``)
and inference (where the caller may have prompted a custom subset).

Implementation notes
--------------------
- Beam search creates ``num_beams × batch_size`` hypotheses.  The FSM
  maintains one state per hypothesis, indexed by the flat hypothesis
  position in the batch.  Tries are stored per *batch item* and shared
  across all beams for that item.
- States are reset at the start of generation and updated token-by-token
  as each new ID is appended to the hypothesis.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence, Set

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from vanilla_s2g.linearisation.special_tokens import SPECIAL_TOKENS, SpecialTokens

logger = logging.getLogger(__name__)


# ===================================================================== #
#                              TRIE                                      #
# ===================================================================== #


class Trie:
    """Prefix tree over pre-tokenised label IDs.

    Each label is supplied as a sequence of subword token IDs and
    inserted into the tree.  At query time, given a sequence of
    already-generated IDs, the trie returns the set of valid next
    tokens.

    The trie also stores one or more *sentinel* token IDs that are valid
    **only** at leaf nodes (i.e. when the label is complete).  For the
    relation trie, the sentinel is ``<tail>``.  For the null trie, the
    sentinels are ``<null>`` and ``EOS``.

    The class no longer takes a tokeniser: callers are expected to
    provide token-ID sequences directly.  In the S2G pipeline these
    sequences are sliced out of the encoder input ``source_ids``, which
    guarantees they coincide with the IDs the model has seen for the
    same labels in the SSI prefix.

    Args:
        label_token_ids:  Iterable of token-ID sequences, one per label.
        sentinel_ids:     Set of token IDs that signal label completion.
    """

    def __init__(
        self,
        label_token_ids: Sequence[Sequence[int]],
        sentinel_ids: Set[int],
    ) -> None:
        self.sentinel_ids: FrozenSet[int] = frozenset(sentinel_ids)

        # Internal tree: each node is a dict mapping token_id → child_node.
        # A special key ``_end`` marks a complete label.
        self._root: Dict = {}

        for ids in label_token_ids:
            if not ids:
                # Defensive: skip empty label spans (e.g. two adjacent
                # <type> markers, which should never occur in practice).
                continue
            node = self._root
            for tid in ids:
                tid_int = int(tid)
                if tid_int not in node:
                    node[tid_int] = {}
                node = node[tid_int]
            node["_end"] = True

    def get_valid_next(self, prefix_ids: List[int]) -> FrozenSet[int]:
        """Return valid next token IDs given the already-generated prefix.

        If the prefix leads to a complete label (leaf node), the sentinel
        IDs are included.  If the prefix is invalid (no matching path),
        only sentinel IDs are returned as a fallback to avoid blocking
        generation entirely.

        Args:
            prefix_ids: Token IDs generated so far for this label.

        Returns:
            Frozen set of valid next token IDs.
        """
        node = self._root

        # Walk down the trie following the prefix.
        for tid in prefix_ids:
            if tid not in node:
                # Prefix fell off the trie — allow sentinels as recovery.
                return self.sentinel_ids
            node = node[tid]

        # Collect valid continuations.
        valid: Set[int] = set()

        # Regular children (continue the label).
        for key in node:
            if key != "_end":
                valid.add(key)

        # If this node is a leaf, the label is complete → allow sentinels.
        if "_end" in node:
            valid.update(self.sentinel_ids)

        return frozenset(valid) if valid else self.sentinel_ids


# ===================================================================== #
#                  EXTRACTING LABELS FROM THE SSI                        #
# ===================================================================== #


def _extract_label_token_ids(
    source_row: List[int],
    type_id: int,
    text_id: int,
    eos_id: int,
    pad_id: int,
) -> List[List[int]]:
    """Slice the per-instance label token-ID spans out of one source row.

    The encoder input has the structure::

        <type> tok* <type> tok* ... <type> tok* <text> ...

    This walker iterates the row left to right and accumulates the token
    IDs that fall between consecutive ``<type>`` markers.  Each completed
    span (i.e. the span ending at the next ``<type>`` or at ``<text>``)
    is appended to the output list.

    Edge cases handled:

    - **Truncation before <text>**: if the row ends (EOS or PAD reached)
      before a ``<text>`` token is seen, any in-progress label is still
      kept on the assumption that it is most likely a complete label
      (the text was clipped, not the label).  Pathological mid-token
      truncation will yield a path the model cannot complete; the trie
      then falls back to its sentinels, which is an acceptable
      degradation.
    - **Empty span between two <type> markers**: the empty span is
      dropped defensively, since an empty label has no meaningful
      decoder behaviour.

    Args:
        source_row:  Token IDs for a single batch item (already a list).
        type_id:     Token ID of ``<type>``.
        text_id:     Token ID of ``<text>``.
        eos_id:      Tokeniser's EOS token ID.
        pad_id:      Tokeniser's pad token ID.

    Returns:
        List of label token-ID sequences, one per prompted relation type.
    """
    labels: List[List[int]] = []
    current: Optional[List[int]] = None  # ``None`` until first <type>.

    for tid in source_row:
        if tid == pad_id or tid == eos_id:
            break
        if tid == type_id:
            if current:                      # close previous span (if any)
                labels.append(current)
            current = []
            continue
        if tid == text_id:
            if current:
                labels.append(current)
            return labels                    # SSI prefix complete, stop
        if current is not None:
            current.append(int(tid))

    # Reached end-of-row without seeing <text>: treat any in-progress
    # span as a (best-effort) complete label.
    if current:
        labels.append(current)
    return labels


# ===================================================================== #
#                           FSM STATES                                   #
# ===================================================================== #


class FSMState(Enum):
    """Decoder FSM states corresponding to the specification."""

    START = auto()
    GENERATE_ENT_SPAN = auto()
    GENERATE_REL_LABEL = auto()
    GENERATE_TAIL_SPAN = auto()
    GENERATE_NULL_LABEL = auto()
    END = auto()


# ===================================================================== #
#                     PER-HYPOTHESIS STATE                               #
# ===================================================================== #


class HypothesisState:
    """Mutable state tracked for a single beam hypothesis.

    Attributes:
        fsm_state:      Current FSM state.
        label_prefix:   Accumulated token IDs for the current label
                        (used as trie prefix in REL and NULL states).
        span_tokens:    Accumulated token IDs for the current span
                        (used for source-copy matching in ENT/TAIL states).
    """

    __slots__ = ("fsm_state", "label_prefix", "span_tokens")

    def __init__(self) -> None:
        self.fsm_state: FSMState = FSMState.START
        self.label_prefix: List[int] = []
        self.span_tokens: List[int] = []


# ===================================================================== #
#                    CONSTRAINT LOGITS PROCESSOR                         #
# ===================================================================== #


class ConstraintDecodingProcessor(LogitsProcessor):
    """HuggingFace ``LogitsProcessor`` that enforces SEL constraints.

    One instance is created per ``generate()`` call and is stateful:
    it tracks the FSM state for every hypothesis in the batch and stores
    one relation trie and one null trie **per batch item** (not
    per hypothesis — beams sharing the same source share the same trie).

    Per-instance tries are constructed in :meth:`__init__` by walking
    each row of ``source_ids`` to extract the relation labels prompted
    in that row's SSI.  The tries restrict the decoder's outputs to
    exactly those labels — i.e. to the types the model was asked about
    in its prompt.

    Args:
        tokenizer:       HuggingFace tokeniser with S2G special tokens.
        source_ids:      Encoder input IDs ``(batch, src_len)``.  Used
                         both for the source-copy constraint and as the
                         source of truth for label trie construction.
        special_tokens:  Token registry.
        num_beams:       Number of beams per batch item.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source_ids: torch.Tensor,
        special_tokens: SpecialTokens,
        num_beams: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.source_ids = source_ids  # (batch, src_len)
        self.num_beams = num_beams

        # Resolve special-token IDs.
        id_of = lambda tok: tokenizer.convert_tokens_to_ids(tok)
        self.type_id: int = id_of(special_tokens.type_prompt)
        self.text_id: int = id_of(special_tokens.text_start)
        self.ent_id: int = id_of(special_tokens.ent_start)
        self.rel_id: int = id_of(special_tokens.rel_open)
        self.tail_id: int = id_of(special_tokens.tail_start)
        self.null_id: int = id_of(special_tokens.reject_type)
        self.eos_id: int = tokenizer.eos_token_id
        self.pad_id: int = tokenizer.pad_token_id or 0

        # Structural tokens that can transition states in ENT/TAIL spans.
        self.span_exit_ids: FrozenSet[int] = frozenset(
            {self.rel_id, self.ent_id, self.null_id, self.eos_id}
        )

        # Pre-compute per-batch-item source token sets for source-copy.
        batch_size = source_ids.shape[0]
        self._source_token_sets: List[FrozenSet[int]] = []
        for b in range(batch_size):
            tokens = set(source_ids[b].tolist())
            tokens.discard(self.pad_id)
            tokens.discard(self.eos_id)
            self._source_token_sets.append(frozenset(tokens))

        # Build per-batch-item label tries from the SSI prefixes.  Each
        # row of source_ids carries the exact token IDs the encoder saw
        # for that instance's prompted relation types, so the tries
        # encode precisely the labels the model was asked about.
        sentinels_rel: Set[int] = {self.tail_id}
        sentinels_null: Set[int] = {self.null_id, self.eos_id}
        self.rel_tries: List[Trie] = []
        self.null_tries: List[Trie] = []
        for b in range(batch_size):
            label_ids = _extract_label_token_ids(
                source_row=source_ids[b].tolist(),
                type_id=self.type_id,
                text_id=self.text_id,
                eos_id=self.eos_id,
                pad_id=self.pad_id,
            )
            self.rel_tries.append(Trie(label_ids, sentinels_rel))
            self.null_tries.append(Trie(label_ids, sentinels_null))

        # Per-hypothesis FSM states.  Initialised lazily on first call
        # when the actual expanded batch size is known.
        self._states: Optional[List[HypothesisState]] = None

    def _init_states(self, total_hypotheses: int) -> None:
        """Lazily initialise one ``HypothesisState`` per hypothesis."""
        self._states = [HypothesisState() for _ in range(total_hypotheses)]

    def _batch_idx(self, hyp_idx: int) -> int:
        """Map a flat hypothesis index to the original batch index."""
        return hyp_idx // self.num_beams

    # ------------------------------------------------------------------ #
    #  Source-copy allowed tokens                                         #
    # ------------------------------------------------------------------ #

    def _source_copy_allowed(
        self,
        batch_idx: int,
        span_tokens: List[int],
    ) -> FrozenSet[int]:
        """Return tokens valid as the next token in a source-copy span.

        Finds all positions in the source where the current *span_tokens*
        match, then returns the set of tokens that could follow at those
        positions.  Structural tokens (``<rel>``, ``<ent>``, ``<null>``,
        ``EOS``) are always included to allow ending the span.

        Args:
            batch_idx:   Index into the original (unexpanded) batch.
            span_tokens: Token IDs generated so far for this span.

        Returns:
            Frozen set of allowed next token IDs.
        """
        src = self.source_ids[batch_idx].tolist()
        n = len(span_tokens)

        if n == 0:
            # Any source token is valid as the first token of a span.
            return frozenset(self._source_token_sets[batch_idx] | self.span_exit_ids)

        # Find positions where span_tokens match in the source.
        valid_next: Set[int] = set()
        for i in range(len(src) - n):
            if src[i : i + n] == span_tokens:
                next_tok = src[i + n]
                if next_tok != self.pad_id:
                    valid_next.add(next_tok)

        # Always allow structural exits.
        valid_next.update(self.span_exit_ids)
        return frozenset(valid_next)

    # ------------------------------------------------------------------ #
    #  Allowed tokens per state                                           #
    # ------------------------------------------------------------------ #

    def _allowed_tokens(self, hyp_idx: int) -> FrozenSet[int]:
        """Compute the set of allowed next tokens for one hypothesis.

        Dispatches to the appropriate constraint mechanism based on the
        current FSM state.
        """
        state = self._states[hyp_idx]  # type: ignore[index]
        batch_idx = self._batch_idx(hyp_idx)

        if state.fsm_state == FSMState.START:
            # Must start with <ent>.
            return frozenset({self.ent_id})

        elif state.fsm_state == FSMState.GENERATE_ENT_SPAN:
            return self._source_copy_allowed(batch_idx, state.span_tokens)

        elif state.fsm_state == FSMState.GENERATE_REL_LABEL:
            return self.rel_tries[batch_idx].get_valid_next(state.label_prefix)

        elif state.fsm_state == FSMState.GENERATE_TAIL_SPAN:
            return self._source_copy_allowed(batch_idx, state.span_tokens)

        elif state.fsm_state == FSMState.GENERATE_NULL_LABEL:
            return self.null_tries[batch_idx].get_valid_next(state.label_prefix)

        elif state.fsm_state == FSMState.END:
            return frozenset({self.eos_id, self.pad_id})

        # Should never reach here.
        return frozenset({self.eos_id})

    # ------------------------------------------------------------------ #
    #  State transition                                                   #
    # ------------------------------------------------------------------ #

    def _transition(self, hyp_idx: int, token_id: int) -> None:
        """Update the FSM state for *hyp_idx* after emitting *token_id*."""
        state = self._states[hyp_idx]  # type: ignore[index]

        if token_id == self.eos_id:
            state.fsm_state = FSMState.END
            return

        if token_id == self.pad_id:
            # Padding after EOS — no transition.
            return

        if token_id == self.ent_id:
            state.fsm_state = FSMState.GENERATE_ENT_SPAN
            state.span_tokens = []
            state.label_prefix = []
            return

        if token_id == self.rel_id:
            state.fsm_state = FSMState.GENERATE_REL_LABEL
            state.label_prefix = []
            state.span_tokens = []
            return

        if token_id == self.tail_id:
            state.fsm_state = FSMState.GENERATE_TAIL_SPAN
            state.span_tokens = []
            return

        if token_id == self.null_id:
            state.fsm_state = FSMState.GENERATE_NULL_LABEL
            state.label_prefix = []
            return

        # Regular content token — accumulate in the relevant buffer.
        if state.fsm_state == FSMState.GENERATE_ENT_SPAN:
            state.span_tokens.append(token_id)
        elif state.fsm_state == FSMState.GENERATE_REL_LABEL:
            state.label_prefix.append(token_id)
        elif state.fsm_state == FSMState.GENERATE_TAIL_SPAN:
            state.span_tokens.append(token_id)
        elif state.fsm_state == FSMState.GENERATE_NULL_LABEL:
            state.label_prefix.append(token_id)

    # ------------------------------------------------------------------ #
    #  LogitsProcessor interface                                          #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply FSM constraints to *scores* for the current decoding step.

        ``input_ids`` has shape ``(batch_size × num_beams, seq_len)``
        where ``seq_len`` grows by 1 at each step.  ``scores`` has shape
        ``(batch_size × num_beams, vocab_size)``.

        At the first call (``seq_len == 1``, the decoder start token),
        states are initialised.  At subsequent calls, the last emitted
        token is used to transition each hypothesis's FSM, then the
        allowed-token set is computed and all disallowed tokens are
        masked to ``-inf``.
        """
        num_hyps = input_ids.shape[0]

        # Lazy initialisation.
        if self._states is None:
            self._init_states(num_hyps)

        # The first column is the decoder start token (usually pad or EOS
        # from the model config).  We only need to transition on tokens
        # the model has actually chosen, which start from column 1.
        seq_len = input_ids.shape[1]
        if seq_len > 1:
            # Transition on the last emitted token.
            last_tokens = input_ids[:, -1].tolist()
            for h in range(num_hyps):
                self._transition(h, last_tokens[h])

        # Compute allowed tokens and apply mask.
        neg_inf = float("-inf")
        for h in range(num_hyps):
            allowed = self._allowed_tokens(h)
            mask = torch.full_like(scores[h], neg_inf)
            for tid in allowed:
                mask[tid] = 0.0
            scores[h] = scores[h] + mask

        return scores


# ===================================================================== #
#                     BUILDER FUNCTION                                   #
# ===================================================================== #


def build_constraint_processor(
    tokenizer: PreTrainedTokenizerBase,
    source_ids: torch.Tensor,
    special_tokens: Optional[SpecialTokens] = None,
    num_beams: int = 1,
) -> ConstraintDecodingProcessor:
    """Construct a :class:`ConstraintDecodingProcessor` ready for use.

    The processor builds its relation and null tries internally, one
    pair per batch item, by extracting the prompted relation types from
    each row of ``source_ids``.  Callers therefore no longer pass an
    explicit list of relation types: the prompt is the source of truth.

    Args:
        tokenizer:       HuggingFace tokeniser with S2G special tokens.
        source_ids:      Encoder input IDs ``(batch, src_len)``.
        special_tokens:  Token registry (defaults to the module singleton).
        num_beams:       Number of beams per batch item.

    Returns:
        A configured :class:`ConstraintDecodingProcessor`.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    return ConstraintDecodingProcessor(
        tokenizer=tokenizer,
        source_ids=source_ids,
        special_tokens=special_tokens,
        num_beams=num_beams,
    )