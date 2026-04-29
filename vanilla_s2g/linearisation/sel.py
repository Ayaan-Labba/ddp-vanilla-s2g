"""
Structured Extraction Language (SEL) construction and parsing.

The SEL is a linearised shallow tree that encodes entities, their pairwise
relations, and explicit rejections of schema types absent from the text.
This module is responsible for both directions of the conversion:

1. **Construction** — Given structured entities, relations, and rejected
   types, produce the flat SEL target string that the decoder is trained
   to generate.

2. **Parsing** — Given a flat token sequence produced by the decoder,
   recover the structured entities, relations, and rejected types, then
   flatten them into ``(head, type, tail)`` triplets for evaluation.

The SEL grammar (from the specification)::

    ENT   ::=  <ent> SPAN REL*
    REL   ::=  <rel> LABEL <tail> SPAN
    LABEL ::=  token+                    (natural-language type name)
    SPAN  ::=  token+                    (surface text copied from input)
    NULL  ::=  <null> LABEL              (absent type)

Design note
-----------
Construction and parsing are co-located in one module so that any change
to the linearisation format is reflected in both directions at once.
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Optional, Set, Tuple

from .special_tokens import SPECIAL_TOKENS, SpecialTokens

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EntityBlock = Dict[str, Any]
# {"text": str, "relations": [{"type": str, "tail": str}]}
# During construction, may also carry "offset" and "type" metadata.

Triplet = Tuple[str, str, str]  # (head_text, relation_type, tail_text)


# ===================================================================== #
#                       SEL CONSTRUCTION                                 #
# ===================================================================== #


def organize_by_entity(
    entities: List[Dict],
    relations: List[Dict],
) -> List[EntityBlock]:
    """Group flat entity and relation lists into entity-centric blocks.

    Each entity becomes a block containing its surface text and a list of
    relations where it serves as the **head**.  Entities that appear only as
    tails still receive a block with an empty relation list.

    Ordering follows the specification: entities are sorted by their offset
    in the source text (start position), and relations within each entity
    are sorted by the tail entity's offset.

    Args:
        entities:  List of entity dicts.  Required keys: ``text``,
                   ``offset`` (``[start, end]``).  Optional: ``type``.
        relations: List of relation dicts.  Required keys: ``head``
                   (entity dict), ``tail`` (entity dict), ``type`` (str).

    Returns:
        List of :data:`EntityBlock` dicts, ordered by text position.
    """
    # Sort entities by start offset.
    sorted_entities = sorted(entities, key=lambda e: e["offset"][0])

    # Build blocks and an offset → index lookup for O(1) relation assignment.
    entity_blocks: List[EntityBlock] = []
    offset_to_idx: Dict[Tuple[int, int], int] = {}

    for entity in sorted_entities:
        offset_key = (entity["offset"][0], entity["offset"][1])
        block: EntityBlock = {
            "text": entity["text"],
            "offset": list(entity["offset"]),
            "type": entity.get("type", ""),
            "relations": [],
        }
        entity_blocks.append(block)
        offset_to_idx[offset_key] = len(entity_blocks) - 1

    # Assign each relation to its head entity's block.
    for rel in relations:
        head_offset = (rel["head"]["offset"][0], rel["head"]["offset"][1])
        if head_offset not in offset_to_idx:
            continue  # Defensive: skip orphaned relations.
        idx = offset_to_idx[head_offset]
        entity_blocks[idx]["relations"].append(
            {
                "type": rel["type"],
                "tail": rel["tail"]["text"],
                "_tail_offset": rel["tail"]["offset"][0],
            }
        )

    # Sort relations within each block by tail offset, then strip helper key.
    for block in entity_blocks:
        block["relations"].sort(key=lambda r: r["_tail_offset"])
        for rel in block["relations"]:
            del rel["_tail_offset"]

    return entity_blocks


def filter_entity_blocks(
    entity_blocks: List[EntityBlock],
    allowed_types: Set[str],
) -> List[EntityBlock]:
    """Remove relation blocks whose type is not in *allowed_types*.

    Entity blocks themselves are **never** removed — only their ``<rel>``
    children are pruned.  This implements the positive-type withholding
    behaviour described in Section 3.1 of the specification: when a
    positive type is withheld from the SSI, the corresponding ``<rel>``
    blocks are dropped but the entities remain.

    Args:
        entity_blocks: Output of :func:`organize_by_entity`.
        allowed_types: Set of relation-type strings to retain.

    Returns:
        A *new* list of entity blocks with filtered relations.
    """
    filtered: List[EntityBlock] = []
    for block in entity_blocks:
        new_block: EntityBlock = {
            "text": block["text"],
            "relations": [
                rel for rel in block["relations"] if rel["type"] in allowed_types
            ],
        }
        # Preserve optional metadata if present.
        if "offset" in block:
            new_block["offset"] = block["offset"]
        if "type" in block:
            new_block["type"] = block["type"]
        filtered.append(new_block)
    return filtered


def build_sel(
    entity_blocks: List[EntityBlock],
    rejected_types: Optional[List[str]] = None,
    random_sel: bool = False,
    special_tokens: Optional[SpecialTokens] = None,
) -> str:
    """Build the flat SEL target string from entity blocks and rejected types.

    Args:
        entity_blocks:  List of :data:`EntityBlock` dicts (typically the
                        output of :func:`organize_by_entity` or
                        :func:`filter_entity_blocks`).
        rejected_types: Relation-type strings to append as ``<null>`` blocks.
        random_sel:     If ``True``, randomise entity order and relation
                        order within each entity.
        special_tokens: Token registry (defaults to the module singleton).

    Returns:
        Flat SEL string ready for tokenisation as the decoder target.

    Example::

        >>> blocks = [
        ...     {"text": "Barack Obama", "relations": [
        ...         {"type": "place of birth", "tail": "Honolulu"},
        ...     ]},
        ...     {"text": "Honolulu", "relations": []},
        ... ]
        >>> build_sel(blocks, rejected_types=["founded"])
        '<ent> Barack Obama <rel> place of birth <tail> Honolulu <ent> Honolulu <null> founded'
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS
    if rejected_types is None:
        rejected_types = []

    st = special_tokens
    blocks = list(entity_blocks)

    if random_sel:
        _random.shuffle(blocks)

    parts: List[str] = []

    for entity in blocks:
        parts.append(f"{st.ent_start} {entity['text']}")

        rels = list(entity["relations"])
        if random_sel:
            _random.shuffle(rels)

        for rel in rels:
            parts.append(f"{st.rel_open} {rel['type']} {st.tail_start} {rel['tail']}")

    # Append null blocks for rejected types.
    null_types = list(rejected_types)
    if random_sel:
        _random.shuffle(null_types)

    for rtype in null_types:
        parts.append(f"{st.reject_type} {rtype}")

    return " ".join(parts)


# ===================================================================== #
#                          SEL PARSING                                   #
# ===================================================================== #


class _State:
    """Parser states for the left-to-right SEL scan."""

    EXPECT_ENT_OR_NULL = "EXPECT_ENT_OR_NULL"
    READ_ENT_SPAN = "READ_ENT_SPAN"
    READ_REL_LABEL = "READ_REL_LABEL"
    READ_TAIL_SPAN = "READ_TAIL_SPAN"
    READ_NULL_LABEL = "READ_NULL_LABEL"


def parse_sel(
    text: str,
    special_tokens: Optional[SpecialTokens] = None,
) -> Tuple[List[EntityBlock], List[str]]:
    """Parse a generated SEL string into structured entities and rejected types.

    Implements the single-pass left-to-right algorithm from Section 6.2 of
    the specification.  The input is the raw decoded string produced by
    ``tokenizer.decode(output_ids, skip_special_tokens=False)``.

    Args:
        text:           Decoded SEL string from the model.
        special_tokens: Token registry (defaults to the module singleton).

    Returns:
        ``(entities, rejected)`` where *entities* is a list of
        :data:`EntityBlock` dicts and *rejected* is a list of
        relation-type strings that the model explicitly rejected.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    st = special_tokens
    special_set: Set[str] = set(st.all_tokens)

    # Forcefully pad special tokens with spaces so glued tokens detach
    padded_text = text
    for token in special_set:
        padded_text = padded_text.replace(token, f" {token} ")

    # Tokenise by whitespace.  Special tokens were added as whole tokens to
    # the vocabulary, so after decode() they appear as space-separated words.
    words = padded_text.strip().split()

    # Merge consecutive non-special words into a single content token so
    # that spans like "Barack Obama" are handled naturally.  Special tokens
    # remain as individual elements.
    tokens: List[str] = _segment_words(words, special_set)

    # ---- Parser state ----
    entities: List[EntityBlock] = []
    rejected: List[str] = []
    current_entity: Optional[EntityBlock] = None
    current_rel_label: Optional[str] = None
    current_tail_span: Optional[str] = None
    state = _State.EXPECT_ENT_OR_NULL

    def flush_tail() -> None:
        nonlocal current_rel_label, current_tail_span
        if (
            current_tail_span is not None
            and current_rel_label is not None
            and current_entity is not None
        ):
            current_entity["relations"].append(
                {
                    "type": current_rel_label.strip(),
                    "tail": current_tail_span.strip(),
                }
            )
        current_rel_label = None
        current_tail_span = None

    def flush_entity() -> None:
        nonlocal current_entity
        if current_entity is not None and current_entity["text"].strip():
            current_entity["text"] = current_entity["text"].strip()
            entities.append(current_entity)
        current_entity = None

    def flush_null() -> None:
        nonlocal current_rel_label
        if current_rel_label is not None and current_rel_label.strip():
            rejected.append(current_rel_label.strip())
        current_rel_label = None

    # ---- Main scan ----
    for token in tokens:
        if token == st.ent_start:
            if state == _State.READ_NULL_LABEL:
                flush_null()
            flush_tail()
            flush_entity()
            current_entity = {"text": "", "relations": []}
            state = _State.READ_ENT_SPAN

        elif token == st.rel_open:
            flush_tail()
            current_rel_label = ""
            state = _State.READ_REL_LABEL

        elif token == st.tail_start:
            current_tail_span = ""
            state = _State.READ_TAIL_SPAN

        elif token == st.reject_type:
            if state == _State.READ_NULL_LABEL:
                flush_null()
            flush_tail()
            flush_entity()
            current_entity = None
            current_rel_label = ""
            state = _State.READ_NULL_LABEL

        else:
            # Regular content token (already merged span).
            if state == _State.READ_ENT_SPAN and current_entity is not None:
                current_entity["text"] = _append(current_entity["text"], token)
            elif state == _State.READ_REL_LABEL and current_rel_label is not None:
                current_rel_label = _append(current_rel_label, token)
            elif state == _State.READ_TAIL_SPAN and current_tail_span is not None:
                current_tail_span = _append(current_tail_span, token)
            elif state == _State.READ_NULL_LABEL and current_rel_label is not None:
                current_rel_label = _append(current_rel_label, token)

    # ---- Flush remaining state ----
    if state == _State.READ_NULL_LABEL:
        flush_null()
    flush_tail()
    flush_entity()

    # ---- Deduplicate entities by text span ----
    deduped = _deduplicate_entities(entities)

    return deduped, rejected


# ===================================================================== #
#                       TRIPLET EXTRACTION                               #
# ===================================================================== #


def extract_triplets(entities: List[EntityBlock]) -> List[Triplet]:
    """Flatten entity blocks into ``(head, relation_type, tail)`` triplets.

    Args:
        entities: Parsed entity blocks (output of :func:`parse_sel`).

    Returns:
        List of triplet tuples.
    """
    triplets: List[Triplet] = []
    for entity in entities:
        for rel in entity["relations"]:
            triplets.append((entity["text"], rel["type"], rel["tail"]))
    return triplets


# ===================================================================== #
#                          HELPERS                                       #
# ===================================================================== #


def _segment_words(words: List[str], special_set: Set[str]) -> List[str]:
    """Merge consecutive non-special words into single span strings.

    Special tokens remain as individual elements.  This avoids the parser
    having to accumulate single words character by character.

    Example::

        ["<ent>", "Barack", "Obama", "<rel>", "place", "of", "birth"]
        → ["<ent>", "Barack Obama", "<rel>", "place of birth"]
    """
    tokens: List[str] = []
    buffer: List[str] = []

    for w in words:
        if w in special_set:
            if buffer:
                tokens.append(" ".join(buffer))
                buffer = []
            tokens.append(w)
        else:
            buffer.append(w)

    if buffer:
        tokens.append(" ".join(buffer))

    return tokens


def _append(current: str, token: str) -> str:
    """Append *token* to *current* with a separating space if needed."""
    if current:
        return f"{current} {token}"
    return token


def _deduplicate_entities(entities: List[EntityBlock]) -> List[EntityBlock]:
    """Merge duplicate entity blocks (same text span) into a single block.

    Relations from duplicate blocks are appended to the first occurrence.
    """
    seen: Dict[str, int] = {}
    deduped: List[EntityBlock] = []

    for ent in entities:
        key = ent["text"]
        if key in seen:
            deduped[seen[key]]["relations"].extend(ent["relations"])
        else:
            seen[key] = len(deduped)
            deduped.append(ent)

    return deduped