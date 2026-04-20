"""
S2G Data Collator — dynamic SSI construction with type sampling.

The collator is the bridge between raw dataset instances and the
tokenised tensors consumed by the model.  For each instance in a batch,
it performs the following steps:

1. **Positive sampling** — each ground-truth relation type is
   independently included with probability ``positive_rate``.  Types
   that are withheld have their ``<rel>`` blocks removed from the
   target SEL (but their entities are retained).

2. **Negative sampling** — up to *k(t)* negative types are selected from
   the full schema, where *k* increases linearly from
   ``negative_max_start`` to ``negative_max_end`` over the course of
   training.  Each selected negative is independently sampled with
   probability ``negative_rate``.  Sampled negatives are appended as
   ``<null>`` blocks in the target SEL.

3. **SSI construction** — the sampled positive and negative types are
   combined into the SSI prefix, and the full encoder input is built.

4. **Tokenisation** — encoder inputs and decoder targets are tokenised,
   padded, and returned as a batch dict ready for the model.

The current training step is maintained by a
:class:`~vanilla_s2g.evaluation.callbacks.StepTrackingCallback` that
sets :attr:`current_step` after each optimiser step.  Because PyTorch
DataLoader workers only run ``Dataset.__getitem__`` — the
``collate_fn`` executes in the main process — step updates are always
visible to the collator.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set

from transformers import PreTrainedTokenizerBase

from vanilla_s2g.linearisation import (
    build_encoder_input,
    build_sel,
    filter_entity_blocks,
    organize_by_entity,
)


class S2GCollator:
    """Data collator with dynamic SSI and type-sampling logic.

    Args:
        tokenizer:  HuggingFace tokeniser with S2G special tokens registered.
        schema:     Complete list of relation-type strings for the dataset.
        config:     Dict-like configuration with the keys documented below.

    Required config keys::

        max_source_length   – int   (encoder token limit)
        max_target_length   – int   (decoder token limit)
        max_steps           – int   (total training steps, for schedule)
        positive_rate       – float (Bernoulli prob. for positive types)
        negative_rate       – float (Bernoulli prob. for selected negatives)
        negative_max_start  – int   (k at step 0)
        negative_max_end    – int   (k at step T)
        random_prompt       – bool  (shuffle SSI type order)
        random_sel          – bool  (shuffle entity/relation order in SEL)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        schema: List[str],
        config: Dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.schema = list(schema)
        self.schema_set: Set[str] = set(schema)
        self.config = config

        # Mutable step counter, updated by StepTrackingCallback.
        self._current_step: int = 0

    # ------------------------------------------------------------------ #
    # Step tracking interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def current_step(self) -> int:
        """Current global training step (set by the callback)."""
        return self._current_step

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._current_step = value

    # ------------------------------------------------------------------ #
    # Negative-cap schedule                                               #
    # ------------------------------------------------------------------ #

    def _compute_neg_cap(self) -> int:
        """Compute the maximum number of negative types at the current step.

        Implements the linear schedule::

            k(t) = k_start + floor((k_end - k_start) * t / T)

        Returns:
            Maximum number of negative types to select from the schema.
        """
        t = self._current_step
        total = self.config["max_steps"]
        k_start = self.config["negative_max_start"]
        k_end = self.config["negative_max_end"]

        if total <= 0:
            return k_end

        progress = min(t / total, 1.0)
        return k_start + int((k_end - k_start) * progress)

    # ------------------------------------------------------------------ #
    # Type sampling                                                       #
    # ------------------------------------------------------------------ #

    def _sample_types(
        self,
        instance_types: List[str],
    ) -> tuple[List[str], List[str]]:
        """Sample positive and negative relation types for one instance.

        **Positive sampling:**  Each type in *instance_types* is
        independently included with probability ``positive_rate``.  If all
        types are withheld (unlikely but possible), one is retained at
        random to avoid a degenerate training signal.

        **Negative sampling:**  Up to *k(t)* types are selected uniformly
        from the schema's negative pool (types not in *instance_types*).
        Each selected type is then independently sampled with probability
        ``negative_rate``.

        Args:
            instance_types: Relation types present in this instance.

        Returns:
            ``(sampled_positives, sampled_negatives)``
        """
        pos_rate = self.config["positive_rate"]
        neg_rate = self.config["negative_rate"]

        # --- Positive sampling ---
        instance_set = set(instance_types)
        sampled_pos = [t for t in instance_types if random.random() < pos_rate]

        # Safety net: keep at least one positive type if any exist.
        if instance_types and not sampled_pos:
            sampled_pos = [random.choice(instance_types)]

        # --- Negative sampling ---
        negative_pool = [t for t in self.schema if t not in instance_set]
        k = min(self._compute_neg_cap(), len(negative_pool))
        selected_negatives = random.sample(negative_pool, k) if k > 0 else []
        sampled_neg = [t for t in selected_negatives if random.random() < neg_rate]

        return sampled_pos, sampled_neg

    # ------------------------------------------------------------------ #
    # Collation                                                           #
    # ------------------------------------------------------------------ #

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate a list of raw instances into a tokenised model batch.

        Args:
            batch: List of instance dicts from :class:`S2GDataset`.

        Returns:
            Dictionary with ``input_ids``, ``attention_mask``, and
            ``labels`` tensors ready for the seq2seq model.
        """
        encoder_inputs: List[str] = []
        decoder_targets: List[str] = []

        for instance in batch:
            encoder_input, decoder_target = self._prepare_instance(instance)
            encoder_inputs.append(encoder_input)
            decoder_targets.append(decoder_target)

        return self._tokenize_batch(encoder_inputs, decoder_targets)

    def _prepare_instance(self, instance: Dict) -> tuple[str, str]:
        """Build the encoder input and decoder target for a single instance.

        Applies positive/negative sampling, entity-block filtering, and
        SSI/SEL construction.

        Args:
            instance: Raw dict from the dataset.

        Returns:
            ``(encoder_input_str, decoder_target_str)``
        """
        instance_types: List[str] = instance["types"]

        # 1. Sample types.
        sampled_pos, sampled_neg = self._sample_types(instance_types)
        sampled_pos_set: Set[str] = set(sampled_pos)

        # 2. Build entity blocks from the instance data and filter
        #    relations to only the sampled positive types.
        entity_blocks = organize_by_entity(
            instance["entities"], instance["relations"]
        )
        filtered_blocks = filter_entity_blocks(entity_blocks, sampled_pos_set)

        # 3. Build SSI (sampled positives + sampled negatives).
        ssi_types = sampled_pos + sampled_neg
        encoder_input = build_encoder_input(
            ssi_types,
            instance["text"],
            random_prompt=self.config["random_prompt"],
        )

        # 4. Build SEL target with null blocks for sampled negatives.
        decoder_target = build_sel(
            filtered_blocks,
            rejected_types=sampled_neg,
            random_sel=self.config["random_sel"],
        )

        return encoder_input, decoder_target

    def _tokenize_batch(
        self,
        encoder_inputs: List[str],
        decoder_targets: List[str],
    ) -> Dict[str, Any]:
        """Tokenise and pad encoder inputs and decoder targets.

        Padding tokens in the labels are replaced with ``-100`` so that
        they are ignored by the cross-entropy loss.

        Args:
            encoder_inputs:  List of SSI + text strings.
            decoder_targets: List of SEL target strings.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``labels``.
        """
        max_src = self.config["max_source_length"]
        max_tgt = self.config["max_target_length"]

        # Tokenise encoder inputs.
        model_inputs = self.tokenizer(
            encoder_inputs,
            max_length=max_src,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        # Tokenise decoder targets.
        labels = self.tokenizer(
            decoder_targets,
            max_length=max_tgt,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        # Replace padding token IDs with -100 for loss masking.
        label_ids = labels["input_ids"].clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids

        return model_inputs