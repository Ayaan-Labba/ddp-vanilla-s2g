"""
Training callbacks for the S2G pipeline.

Provides HuggingFace Trainer-compatible callbacks for:

1. **StepTrackingCallback** — Synchronises the data collator's
   ``current_step`` counter with the Trainer's global step after each
   optimiser update.  This drives the negative-type cap schedule.

2. **GenerateTextSamplesCallback** — Periodically runs inference on
   a held-out sample batch, parses the SEL output, and logs a W&B
   table comparing source text, predicted triplets, and gold triplets.

3. **PeriodicCheckpointCallback** — Saves a full resumable checkpoint
   at a fixed step interval, independent of the validation-metric-based
   top-k checkpoints.  Provides a safety net against SSH disconnections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from vanilla_s2g.linearisation import extract_triplets, parse_sel

logger = logging.getLogger(__name__)


# ===================================================================== #
#                     STEP TRACKING CALLBACK                             #
# ===================================================================== #


class StepTrackingCallback(TrainerCallback):
    """Synchronise the collator's step counter with the Trainer.

    At each training step, the Trainer's ``state.global_step`` is written
    to the collator's ``current_step`` property.  Because the collator's
    ``__call__`` executes in the main process (not in DataLoader workers),
    this update is immediately visible for the next batch.

    Args:
        collator: The :class:`~vanilla_s2g.data.collator.S2GCollator`
                  instance used by the Trainer.
    """

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.collator.current_step = state.global_step


# ===================================================================== #
#                  GENERATE TEXT SAMPLES CALLBACK                        #
# ===================================================================== #


class GenerateTextSamplesCallback(TrainerCallback):
    """Log predicted-vs-gold triplet comparisons to W&B at regular intervals.

    Every *interval* global steps, this callback:

    1. Takes a small sample batch from the validation set.
    2. Runs autoregressive generation (beam search).
    3. Parses the generated SEL into triplets.
    4. Parses the gold SEL into triplets.
    5. Logs a W&B table with columns: source text, predicted triplets,
       gold triplets, predicted SEL, gold SEL.

    This provides a qualitative view of model progress alongside the
    quantitative metrics.

    Args:
        tokenizer:     HuggingFace tokeniser with S2G special tokens.
        sample_batch:  A list of raw instances from the validation set
                       to use as the fixed sample.
        collator:      The collator (for tokenising the sample batch).
        interval:      Log every *interval* global steps.
        eval_beams:    Number of beams for generation.
        max_target_length: Maximum generation length.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sample_batch: List[Dict],
        collator: Any,
        interval: int = 10_000,
        eval_beams: int = 3,
        max_target_length: int = 150,
    ) -> None:
        self.tokenizer = tokenizer
        self.sample_batch = sample_batch
        self.collator = collator
        self.interval = interval
        self.eval_beams = eval_beams
        self.max_target_length = max_target_length
        self._last_logged_step: int = -1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step == 0 or step == self._last_logged_step:
            return
        if step % self.interval != 0:
            return

        self._last_logged_step = step

        if model is None:
            logger.warning(
                "GenerateTextSamplesCallback: model not available at step %d.", step
            )
            return

        try:
            self._log_samples(model, state)
        except Exception:
            logger.exception("GenerateTextSamplesCallback failed at step %d.", step)

    def _log_samples(self, model: Any, state: TrainerState) -> None:
        """Run inference on the sample batch and log to W&B."""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed; skipping sample generation log.")
            return

        if wandb.run is None:
            return

        # Tokenise the sample batch using the collator.
        batch = self.collator(self.sample_batch)
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Generate predictions.
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.eval_beams,
                max_length=self.max_target_length,
                length_penalty=0.0,
                no_repeat_ngram_size=0,
                early_stopping=False,
            )
        model.train()

        # Decode and parse.
        rows: List[List[str]] = []
        for i in range(len(self.sample_batch)):
            # Source text.
            source_text = self.sample_batch[i]["text"]

            # Predicted SEL.
            pred_sel = self.tokenizer.decode(
                generated_ids[i], skip_special_tokens=False
            )
            pred_sel = _clean_decoded(pred_sel, self.tokenizer)
            pred_ents, pred_rej = parse_sel(pred_sel)
            pred_triplets = extract_triplets(pred_ents)

            # Gold SEL.
            gold_ids = labels[i].clone()
            gold_ids[gold_ids == -100] = self.tokenizer.pad_token_id
            gold_sel = self.tokenizer.decode(gold_ids, skip_special_tokens=False)
            gold_sel = _clean_decoded(gold_sel, self.tokenizer)
            gold_ents, gold_rej = parse_sel(gold_sel)
            gold_triplets = extract_triplets(gold_ents)

            rows.append([
                source_text,
                _format_triplets(pred_triplets),
                _format_triplets(gold_triplets),
                pred_sel.strip(),
                gold_sel.strip(),
            ])

        table = wandb.Table(
            columns=["Source Text", "Predicted Triplets", "Gold Triplets",
                      "Predicted SEL", "Gold SEL"],
            data=rows,
        )
        wandb.log({"sample_predictions": table}, step=state.global_step)
        logger.info(
            "Logged %d sample predictions to W&B at step %d.",
            len(rows), state.global_step,
        )


# ===================================================================== #
#                  PERIODIC CHECKPOINT CALLBACK                          #
# ===================================================================== #


class PeriodicCheckpointCallback(TrainerCallback):
    """Save a full resumable checkpoint at a fixed step interval.

    This is a safety net for training interruptions (e.g. SSH
    disconnection).  The checkpoint includes model weights, optimiser
    state, scheduler, RNG states, and the W&B run ID.

    Only one periodic checkpoint is kept at a time (overwritten each
    interval) to avoid excessive disk usage.  This is separate from
    the top-k metric-based checkpoints managed by the Trainer.

    Args:
        output_dir:     Base output directory for checkpoints.
        every_n_steps:  Save every *every_n_steps* global steps.
        wandb_run_id:   W&B run ID to persist for resumption (optional).
    """

    def __init__(
        self,
        output_dir: str,
        every_n_steps: int = 5000,
        wandb_run_id: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.wandb_run_id = wandb_run_id
        self._last_saved_step: int = -1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step == 0 or step == self._last_saved_step:
            return
        if step % self.every_n_steps != 0:
            return

        self._last_saved_step = step

        # Signal the Trainer to save a checkpoint.
        control.should_save = True

        # Persist W&B run ID for seamless resumption.
        if self.wandb_run_id:
            self._save_run_metadata(step)

    def _save_run_metadata(self, step: int) -> None:
        """Write a small JSON file with the W&B run ID and step."""
        meta_path = self.output_dir / "run_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "wandb_run_id": self.wandb_run_id,
            "last_step": step,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


# ===================================================================== #
#                          HELPERS                                       #
# ===================================================================== #


def _clean_decoded(text: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """Strip decoder artefacts from a decoded SEL string.

    Removes the decoder start token, EOS token, and padding tokens that
    HuggingFace inserts.  Normalises whitespace.
    """
    # Common artefacts: <pad>, </s>, leading whitespace.
    for tok in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]:
        if tok:
            text = text.replace(tok, "")
    return " ".join(text.split())


def _format_triplets(
    triplets: List[tuple],
) -> str:
    """Format a list of triplets as a readable multi-line string."""
    if not triplets:
        return "(none)"
    lines = []
    for t in triplets:
        lines.append(f"({t[0]}, {t[1]}, {t[2]})")
    return "\n".join(lines)


def load_run_metadata(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load the W&B run metadata file if it exists.

    Used during training resumption to recover the W&B run ID.

    Args:
        output_dir: Base output directory.

    Returns:
        Metadata dict or ``None`` if the file does not exist.
    """
    meta_path = Path(output_dir) / "run_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)