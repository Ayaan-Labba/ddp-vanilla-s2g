"""
S2G Data Collator — dynamic SSI construction with type sampling.

The collator is the bridge between raw dataset instances and the
tokenised tensors consumed by the model.  Two modes are supported:

- **``schedule`` mode** (pretraining): linear schedules for the positive
  rate, negative rate, and negative cap k(t).  Each schedule
  short-circuits to its end value when start equals end, so a constant
  rate does not pay the per-batch interpolation cost.  ``max_types_in_prompt``,
  when set, acts as a hard upper bound on the SSI prompt and clamps the
  k(t) cap to ``max_types_in_prompt - num_pos_types`` whenever the
  schedule would otherwise overflow the prompt budget.

- **``budget`` mode** (validation / fine-tuning / final evaluation):
  every gold positive is retained and the SSI is filled with uniformly
  sampled negatives up to ``max_types_in_prompt``.  No rate fields are
  read; budget mode is step-independent and exactly mirrors the
  inference-time SSI construction in ``evaluate.py``.

The current training step is maintained by a
:class:`~vanilla_s2g.evaluation.callbacks.StepTrackingCallback` that
sets :attr:`current_step` after each optimiser step.  Step sharing is
retained for any external use, but the eval collator no longer needs
it (budget mode does not consult the step counter).
"""

from __future__ import annotations

import random
import multiprocessing
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

    Required config keys (both modes)::

        mode                  – "schedule" | "budget"
        max_source_length     – int
        max_target_length     – int
        random_prompt         – bool
        random_sel            – bool

    Required in ``schedule`` mode::

        max_steps             – int
        positive_rate_start   – float
        positive_rate_end     – float
        negative_rate_start   – float
        negative_rate_end     – float
        negative_max_start    – int
        negative_max_end      – int
        max_types_in_prompt   – int or None  (None = no prompt-size cap)

    Required in ``budget`` mode::

        max_types_in_prompt   – int          (must be set)
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

        self.mode = config.get("mode", "schedule")
        if self.mode not in ("schedule", "budget"):
            raise ValueError(
                f"Unknown collator mode '{self.mode}'. "
                "Expected 'schedule' or 'budget'."
            )
        if self.mode == "budget" and config.get("max_types_in_prompt") is None:
            raise ValueError(
                "Budget mode requires 'max_types_in_prompt' to be set."
            )

        # Mutable step counter, updated by StepTrackingCallback.
        self._current_step = multiprocessing.Value("i", 0)
        # Optional reference to another collator's step (for eval sharing).
        self._step_source: Optional["S2GCollator"] = None

    # ------------------------------------------------------------------ #
    # Step tracking interface                                             #
    # ------------------------------------------------------------------ #

    @property
    def current_step(self) -> int:
        """Current global training step (read from source if shared)."""
        if self._step_source is not None:
            return self._step_source.current_step
        return self._current_step.value

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._current_step.value = value

    def share_step_with(self, source: "S2GCollator") -> None:
        """Link this collator's step counter to *source*."""
        self._current_step = source._current_step

    # ------------------------------------------------------------------ #
    # Linear schedules                                                    #
    # ------------------------------------------------------------------ #

    def _progress(self) -> float:
        """Linear progress in [0, 1] across training."""
        total = self.config["max_steps"]
        if total <= 0:
            return 1.0
        return min(self.current_step / total, 1.0)

    def _compute_pos_rate(self, progress: float) -> float:
        """Linear schedule for ``positive_rate``; short-circuits when flat."""
        s = self.config["positive_rate_start"]
        e = self.config["positive_rate_end"]
        if s == e:
            return e
        return s + (e - s) * progress

    def _compute_neg_rate(self, progress: float) -> float:
        """Linear schedule for ``negative_rate``; short-circuits when flat."""
        s = self.config["negative_rate_start"]
        e = self.config["negative_rate_end"]
        if s == e:
            return e
        return s + (e - s) * progress

    def _compute_neg_cap(self, progress: float) -> int:
        """Linear schedule k(t) for the negative cap; short-circuits when flat."""
        k_start = self.config["negative_max_start"]
        k_end = self.config["negative_max_end"]
        if k_start == k_end:
            return k_end
        return k_start + int((k_end - k_start) * progress)

    # ------------------------------------------------------------------ #
    # Type sampling                                                       #
    # ------------------------------------------------------------------ #

    def _sample_types(
        self,
        instance_types: List[str],
    ) -> tuple[List[str], List[str]]:
        """Dispatch to the active mode's sampler."""
        if self.mode == "budget":
            return self._sample_types_budget(instance_types)
        return self._sample_types_schedule(instance_types)

    def _sample_types_schedule(
        self,
        instance_types: List[str],
    ) -> tuple[List[str], List[str]]:
        """Schedule-mode sampling for pretraining.

        Computes the three schedules at the current step (each may be
        constant), then:

        1. Bernoulli-samples positives at ``pos_rate``.  If all positives
           are withheld, retains one at random to avoid a degenerate
           training signal.
        2. Computes the effective negative cap as
           ``min(k(t), max_types_in_prompt - num_pos_types)`` when
           ``max_types_in_prompt`` is set, else ``k(t)``.  This is the
           "k(t) > max_types_in_prompt" clamp described in the spec:
           when the k(t) schedule would overflow the prompt budget, the
           negative count is capped at the remaining budget.
        3. Draws that many negatives uniformly from the pool, then
           Bernoulli sub-samples them at ``neg_rate``.
        """
        progress = self._progress()
        pos_rate = self._compute_pos_rate(progress)
        neg_rate = self._compute_neg_rate(progress)
        neg_max = self._compute_neg_cap(progress)
        max_types = self.config.get("max_types_in_prompt", None)

        # --- Positive sampling ---
        instance_set = set(instance_types)
        sampled_pos = [t for t in instance_types if random.random() < pos_rate]
        if instance_types and not sampled_pos:
            sampled_pos = [random.choice(instance_types)]

        # --- Negative cap, with budget clamp ---
        if max_types is not None:
            budget_remaining = max(0, max_types - len(sampled_pos))
            neg_max = min(neg_max, budget_remaining)

        # --- Negative sampling ---
        negative_pool = [t for t in self.schema if t not in instance_set]
        k = min(neg_max, len(negative_pool))
        selected_negatives = random.sample(negative_pool, k) if k > 0 else []
        sampled_neg = [t for t in selected_negatives if random.random() < neg_rate]

        return sampled_pos, sampled_neg

    def _sample_types_budget(
        self,
        instance_types: List[str],
    ) -> tuple[List[str], List[str]]:
        """Budget-mode sampling, mirroring the test-time setup.

        Includes every gold positive and fills the SSI with negatives
        uniformly drawn from the pool up to ``max_types_in_prompt``.
        Positives are never truncated; if they alone meet or exceed the
        budget, no negatives are added.
        """
        max_types = self.config["max_types_in_prompt"]
        instance_set = set(instance_types)
        negative_pool = [t for t in self.schema if t not in instance_set]

        neg_budget = max(0, max_types - len(instance_types))
        n_neg = min(neg_budget, len(negative_pool))
        sampled_neg = random.sample(negative_pool, n_neg) if n_neg > 0 else []

        return list(instance_types), sampled_neg

    # ------------------------------------------------------------------ #
    # Collation                                                           #
    # ------------------------------------------------------------------ #

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate a list of raw instances into a tokenised model batch."""
        encoder_inputs: List[str] = []
        decoder_targets: List[str] = []

        for instance in batch:
            encoder_input, decoder_target = self._prepare_instance(instance)
            encoder_inputs.append(encoder_input)
            decoder_targets.append(decoder_target)

        return self._tokenize_batch(encoder_inputs, decoder_targets)

    def _prepare_instance(self, instance: Dict) -> tuple[str, str]:
        """Build the encoder input and decoder target for a single instance."""
        instance_types: List[str] = instance["types"]

        sampled_pos, sampled_neg = self._sample_types(instance_types)
        sampled_pos_set: Set[str] = set(sampled_pos)

        entity_blocks = organize_by_entity(
            instance["entities"], instance["relations"]
        )
        filtered_blocks = filter_entity_blocks(entity_blocks, sampled_pos_set)

        ssi_types = sampled_pos + sampled_neg
        encoder_input = build_encoder_input(
            ssi_types,
            instance["text"],
            random_prompt=self.config["random_prompt"],
        )

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
        """Tokenise and pad encoder inputs and decoder targets."""
        max_src = self.config["max_source_length"]
        max_tgt = self.config["max_target_length"]

        model_inputs = self.tokenizer(
            encoder_inputs,
            max_length=max_src,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        labels = self.tokenizer(
            decoder_targets,
            max_length=max_tgt,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        label_ids = labels["input_ids"].clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids

        return model_inputs