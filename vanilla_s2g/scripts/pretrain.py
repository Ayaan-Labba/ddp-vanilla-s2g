"""
Pre-training script for the Vanilla S2G model.

Trains a Flan-T5 model on the REBEL dataset using dynamic SSI
construction, progressive negative-type sampling, and SEL linearisation.

Usage::

    # Fresh start
    python -m vanilla_s2g.scripts.pretrain --config configs/pretrain.yaml

    # Resume after interruption (dotlist override)
    python -m vanilla_s2g.scripts.pretrain --config configs/pretrain.yaml \\
        checkpoint.resume_from=outputs/pretrain/checkpoint-last

    # W&B sweep agent (overrides injected automatically as dotlist)
    wandb agent <sweep_id>

    # Manual override (dotted keys for nested fields)
    python -m vanilla_s2g.scripts.pretrain --config configs/pretrain.yaml \\
        optimizer.lr=3e-5 train.batch_size=8 hardware.gpu_ids=[0,1]
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset

from vanilla_s2g.data import S2GCollator, S2GDataset
from vanilla_s2g.evaluation import (
    GenerateTextSamplesCallback,
    PeriodicCheckpointCallback,
    StepTrackingCallback,
    compute_metrics as eval_compute_metrics,
    load_run_metadata,
)
from vanilla_s2g.linearisation import (
    add_special_tokens_to_tokenizer,
    extract_triplets,
    parse_sel,
)
from vanilla_s2g.scripts.config_utils import load_config, load_schema

logger = logging.getLogger(__name__)


# ===================================================================== #
#                     CUSTOM TRAINER SUBCLASS                            #
# ===================================================================== #


class S2GTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with inverse square root scheduler support.

    Overrides ``create_scheduler`` to implement the inverse sqrt schedule
    described in the specification::

        eval_collator: Separate collator for evaluation
        warmup:  lr_t = lr × (t / warmup_steps)
        decay:   lr_t = lr × sqrt(warmup_steps / t)

    All other Trainer behaviour is inherited unchanged.
    """

    def __init__(
            self, 
            eval_collator=None, 
            scheduler_type: str = "inverse_sqrt",
            train_eval_dataset=None,
            **kwargs: Any) -> None:
        self._scheduler_type = scheduler_type
        self.eval_collator = eval_collator
        self.train_eval_dataset = train_eval_dataset
        super().__init__(**kwargs)

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Create the inverse square root LR scheduler."""
        if self.lr_scheduler is not None:
            return
        
        if self._scheduler_type != "inverse_sqrt":
            super().create_scheduler(num_training_steps, optimizer)
            return

        opt = optimizer or self.optimizer
        # Read the warmup steps natively from Hugging Face's TrainingArguments
        warmup = self.args.get_warmup_steps(num_training_steps)

        def lr_lambda(current_step: int) -> float:
            current_step = max(current_step, 1)
            if current_step < warmup:
                return current_step / max(warmup, 1)
            return math.sqrt(warmup / current_step)

        self.lr_scheduler = LambdaLR(opt, lr_lambda)

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_collator is None:
            return super().get_eval_dataloader(eval_dataset)

        original_collator = self.data_collator
        self.data_collator = self.eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate on the val set and, when configured, also on a fixed train subsample.

        The train-subsample pass runs only when this method is invoked with
        the default ``metric_key_prefix='eval'`` (i.e. by the standard HF
        Trainer validation cycle, or by an explicit external call with the
        default prefix).  Metrics from the train subsample are prefixed
        ``train_eval_`` so they coexist with the ``eval_`` metrics in
        ``state.log_history`` and W&B without colliding with
        ``metric_for_best_model='boundary_f1'`` (which HF resolves to
        ``eval_boundary_f1`` for the early-stopping callback).
        """
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if (
            self.train_eval_dataset is not None
            and metric_key_prefix == "eval"
        ):
            train_metrics = self._evaluate_train_subsample(ignore_keys=ignore_keys)
            metrics.update(train_metrics)

        return metrics

    def _evaluate_train_subsample(
        self,
        ignore_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run the eval loop on the fixed train subsample, bypassing ``on_evaluate``.

        Uses the eval collator (budget-mode SSI) through
        ``get_eval_dataloader``, making the train/val F1 gap directly
        interpretable.  ``self.log`` is called so that metrics reach W&B
        and ``state.log_history``, but ``on_evaluate`` callbacks are
        deliberately *not* fired — this prevents ``EarlyStoppingCallback``
        from emitting a "metric not found" warning for the val-only
        ``eval_boundary_f1`` key, and avoids the early-stopping patience
        counter from being touched by a train-side metric.
        """
        eval_dataloader = self.get_eval_dataloader(self.train_eval_dataset)

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )

        output = eval_loop(
            eval_dataloader,
            description="Train-subsample evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="train_eval",
        )

        self.log(output.metrics)
        return output.metrics


# ===================================================================== #
#                     METRICS FUNCTION                                   #
# ===================================================================== #


def make_compute_metrics(tokenizer):
    """Build a compute_metrics function for the Seq2SeqTrainer.

    The returned function receives an ``EvalPrediction`` object with
    ``predictions`` (generated token IDs) and ``label_ids`` (gold token
    IDs), decodes both, parses the SEL, extracts triplets, and computes
    corpus-level boundary F1.

    Args:
        tokenizer: HuggingFace tokeniser with S2G special tokens.

    Returns:
        A callable compatible with the Trainer's ``compute_metrics`` API.
    """
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Replace -100 (loss-masked positions) with pad_token_id for decoding.
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions == -100, pad_id, predictions)
        label_ids = np.where(label_ids == -100, pad_id, label_ids)

        # Decode to strings.
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        gold_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

        # Parse SEL and extract triplets.
        all_pred_triplets: List[list] = []
        all_gold_triplets: List[list] = []
        all_pred_entities: List[list] = []
        all_gold_entities: List[list] = []

        for pred_text, gold_text in zip(pred_texts, gold_texts):
            pred_text = _clean_decoded(pred_text, tokenizer)
            gold_text = _clean_decoded(gold_text, tokenizer)

            pred_ents, _ = parse_sel(pred_text)
            gold_ents, _ = parse_sel(gold_text)

            all_pred_triplets.append(extract_triplets(pred_ents))
            all_gold_triplets.append(extract_triplets(gold_ents))
            all_pred_entities.append([e["text"] for e in pred_ents])
            all_gold_entities.append([e["text"] for e in gold_ents])

        return eval_compute_metrics(
            all_pred_triplets, all_gold_triplets,
            all_pred_entities, all_gold_entities,
            mode="boundary",
        )

    return compute_metrics


def _clean_decoded(text: str, tokenizer) -> str:
    """Strip decoder artefacts from a decoded SEL string."""
    for tok in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]:
        if tok:
            text = text.replace(tok, "")
    return " ".join(text.split())


# ===================================================================== #
#                            MAIN                                        #
# ===================================================================== #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # ---- 1. Load configuration ----
    cfg = load_config()
    logger.info("Configuration loaded: %s", cfg.config_path)

    # ---- 2. GPU and seed setup ----
    # The schema enforces ``gpu_ids`` as Optional[List[int]], so the
    # int-vs-list coercion the previous version performed is no longer
    # necessary: the user must write ``hardware.gpu_ids=[0]`` rather
    # than ``hardware.gpu_ids=0`` (and OmegaConf would error otherwise).
    if cfg.hardware.gpu_ids is not None:
        gpu_str = ",".join(str(g) for g in cfg.hardware.gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logger.info("CUDA_VISIBLE_DEVICES set to: %s", gpu_str)

    set_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    logger.info("Random seed set to %d", cfg.train.seed)

    # ---- 3. W&B initialisation ----
    wandb_run_id: Optional[str] = None
    wandb_resume: Optional[str] = None
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from = cfg.checkpoint.resume_from
    if resume_from is not None:
        # Recover W&B run ID for seamless continuation.
        meta = load_run_metadata(cfg.data.output_dir)
        if meta and meta.get("wandb_run_id"):
            wandb_run_id = meta["wandb_run_id"]
            wandb_resume = "must"
            logger.info("Resuming W&B run: %s", wandb_run_id)

    # Initialise wandb
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        id=wandb_run_id,
        resume=wandb_resume
    )

    # ---- 4. Load data and schema ----
    schema = load_schema(cfg.data.schema_file)
    logger.info("Loaded schema with %d relation types.", len(schema))

    train_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "train.jsonl",
        seed=cfg.train.seed,
    )
    val_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "val.jsonl",
        subset_fraction=cfg.validation.percent_check,
        seed=cfg.train.seed,
    )
    logger.info(
        "Train: %d instances, Val: %d instances",
        len(train_dataset), len(val_dataset),
    )

    # Fixed train subsample for periodic train-metric computation.  Drawn
    # once at startup with the global seed and wrapped in a Subset so we do
    # not re-read train.jsonl from disk.  Evaluated under budget-mode SSI
    # (same as val), making the train/val F1 gap directly comparable.
    train_eval_dataset = None
    if cfg.validation.train_eval_percent_check and cfg.validation.train_eval_percent_check > 0:
        n_train_eval = max(
            1,
            int(len(train_dataset) * cfg.validation.train_eval_percent_check),
        )
        train_eval_indices = rng.choice(
            len(train_dataset), size=n_train_eval, replace=False
        ).tolist()
        train_eval_dataset = Subset(train_dataset, train_eval_indices)
        logger.info(
            "Train-eval subsample: %d instances (%.2f%% of train)",
            len(train_eval_dataset),
            cfg.validation.train_eval_percent_check * 100,
        )

    # ---- 5. Initialise model and tokeniser ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)
    num_added = add_special_tokens_to_tokenizer(tokenizer, model)
    logger.info(
        "Model loaded: %s (%d special tokens added)",
        cfg.model.name, num_added,
    )

    # ---- 6. Create collators (train + eval) ----
    # The collator API takes a flat dict, so we project the relevant
    # nested fields into the shape it expects.  Doing this explicitly
    # keeps the boundary between "config schema" and "internal data
    # structure" clear.

    # Train collator: schedule mode.  Linear schedules for pos_rate,
    # neg_rate, and k(t); max_types_in_prompt clamps k(t) when the
    # schedule would overflow the prompt budget.
    train_collator_config = {
        "mode": "schedule",
        "max_source_length": cfg.tokenization.max_source_length,
        "max_target_length": cfg.tokenization.max_target_length,
        "max_steps": cfg.train.max_steps,
        "positive_rate_start": cfg.ssi.positive_rate_start,
        "positive_rate_end": cfg.ssi.positive_rate_end,
        "negative_rate_start": cfg.ssi.negative_rate_start,
        "negative_rate_end": cfg.ssi.negative_rate_end,
        "negative_max_start": cfg.ssi.negative_max_start,
        "negative_max_end": cfg.ssi.negative_max_end,
        "max_types_in_prompt": cfg.ssi.max_types_in_prompt,
        "random_prompt": cfg.ssi.random_prompt,
        "random_sel": cfg.ssi.random_sel,
    }
    train_collator = S2GCollator(tokenizer, schema, train_collator_config)

    # Eval collator: budget mode, exactly mirroring evaluate.py.  All
    # gold positives included, negatives uniformly drawn to fill
    # max_types_in_prompt.  Step-independent, so no share_step_with.
    eval_collator_config = {
        "mode": "budget",
        "max_source_length": cfg.tokenization.max_source_length,
        "max_target_length": cfg.tokenization.max_target_length,
        "max_types_in_prompt": cfg.ssi.max_types_in_prompt,
        "random_prompt": cfg.ssi.random_prompt,
        "random_sel": cfg.ssi.random_sel,
    }
    eval_collator = S2GCollator(tokenizer, schema, eval_collator_config)

    # ---- 7. Set up callbacks ----
    callbacks = []

    # Step tracking for dynamic negative cap schedule.
    step_tracker = StepTrackingCallback(train_collator)
    callbacks.append(step_tracker)

    # Early stopping on validation F1.
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=cfg.validation.early_stopping_patience,
        )
    )

    # Periodic safety-net checkpoints.
    periodic_ckpt = PeriodicCheckpointCallback(
        output_dir=cfg.data.output_dir,
        every_n_steps=cfg.checkpoint.every_n_steps,
        wandb_run_id=wandb.run.id,
    )
    callbacks.append(periodic_ckpt)

    # Sample generation table for W&B.
    sample_size = min(8, len(val_dataset))
    sample_batch = [val_dataset[i] for i in range(sample_size)]
    gen_samples_cb = GenerateTextSamplesCallback(
        tokenizer=tokenizer,
        sample_batch=sample_batch,
        collator=eval_collator,
        interval=cfg.callbacks.sample_generation_interval,
        eval_beams=cfg.generation.num_beams,
        max_target_length=cfg.tokenization.max_target_length,
    )
    callbacks.append(gen_samples_cb)

    # ---- 8. Configure TrainingArguments ----
    num_gpus = (
        len(cfg.hardware.gpu_ids)
        if cfg.hardware.gpu_ids
        else max(torch.cuda.device_count(), 1)
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.data.output_dir,

        # Training loop.
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_acc_steps,
        max_grad_norm=cfg.train.gradient_clip_value,
        fp16=(cfg.train.precision == "16"),
        bf16=(cfg.train.precision == "bf16"),
        dataloader_num_workers=cfg.hardware.num_workers,
        dataloader_persistent_workers=cfg.hardware.persistent_workers,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,

        # Optimiser (Trainer creates AdamW internally).
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        adam_epsilon=cfg.optimizer.adam_epsilon,

        # Scheduler: handled by S2GTrainer.create_scheduler override.
        # Set to "constant" to prevent Trainer from creating its own.
        warmup_steps=cfg.scheduler.warmup_steps,
        lr_scheduler_type=cfg.scheduler.type if cfg.scheduler.type != "inverse_sqrt" else "constant",

        # Evaluation.
        eval_strategy="steps",
        eval_steps=cfg.validation.check_interval,
        per_device_eval_batch_size=cfg.validation.batch_size,
        predict_with_generate=True,
        generation_max_length=cfg.tokenization.max_target_length,
        generation_num_beams=cfg.generation.num_beams,

        # Checkpointing.
        save_strategy="steps",
        save_steps=cfg.validation.check_interval,
        save_total_limit=cfg.checkpoint.save_top_k + 1,
        load_best_model_at_end=True,
        metric_for_best_model="boundary_f1",
        greater_is_better=True,

        # Logging.
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        run_name=cfg.wandb.run_name,

        # Misc.
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # ---- 9. Create Trainer ----
    trainer = S2GTrainer(
        scheduler_type=cfg.scheduler.type,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_eval_dataset=train_eval_dataset,
        data_collator=train_collator,
        eval_collator=eval_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    # ---- 11. Train ----
    logger.info("Starting pre-training...")
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Pre-training complete.")

    # ---- 12. Save best model ----
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Best model saved to %s", best_dir)

    # ---- 13. Final evaluation on validation set ----
    logger.info("Loading full validation set for final evaluation...")
    
    # Re-load the dataset without a subset_fraction (or set it to 1.0)
    full_val_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "val.jsonl",
        seed=cfg.train.seed, 
    )
    logger.info("Full Val: %d instances", len(full_val_dataset))

    logger.info("Running final evaluation on full validation set...")
    
    # Pass the full dataset directly into the evaluate method
    val_metrics = trainer.evaluate(eval_dataset=full_val_dataset)
    
    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info("Validation metrics saved to %s", metrics_path)
    logger.info("Final val metrics: %s", val_metrics)


if __name__ == "__main__":
    main()