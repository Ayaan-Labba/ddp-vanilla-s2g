"""
Pre-training script for the Vanilla S2G model.

Trains a Flan-T5 Large model on the REBEL dataset using dynamic SSI
construction, progressive negative-type sampling, and SEL linearisation.

Usage::

    # Fresh start
    python -m vanilla_s2g.scripts.pretrain --config configs/pretrain.yaml

    # Resume after interruption
    python -m vanilla_s2g.scripts.pretrain --config configs/pretrain.yaml \\
        --resume_from outputs/pretrain/checkpoint-last

    # W&B sweep agent (overrides injected automatically)
    wandb agent <sweep_id>

    # Manual override
    python -m vanilla_s2g.scripts.pretrain --config configs/pretrain.yaml \\
        --lr 3e-5 --train_batch_size 8
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from torch.optim.lr_scheduler import LambdaLR

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

        warmup:  lr_t = lr × (t / warmup_steps)
        decay:   lr_t = lr × sqrt(warmup_steps / t)

    All other Trainer behaviour is inherited unchanged.
    """

    def __init__(self, warmup_steps: int = 1000, **kwargs: Any) -> None:
        self._warmup_steps = warmup_steps
        super().__init__(**kwargs)

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Create the inverse square root LR scheduler."""
        if self.lr_scheduler is not None:
            return

        opt = optimizer or self.optimizer
        warmup = self._warmup_steps

        def lr_lambda(current_step: int) -> float:
            current_step = max(current_step, 1)
            if current_step < warmup:
                return current_step / max(warmup, 1)
            return math.sqrt(warmup / current_step)

        self.lr_scheduler = LambdaLR(opt, lr_lambda)


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

        metrics = eval_compute_metrics(
            all_pred_triplets, all_gold_triplets,
            all_pred_entities, all_gold_entities,
            mode="boundary",
        )
        return metrics

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
    if hasattr(cfg, "gpu_ids") and cfg.gpu_ids is not None:
        # Normalise gpu_ids: a single int (from --gpu_ids 0) becomes [0].
        if isinstance(cfg.gpu_ids, (int, float)):
            cfg.gpu_ids = [int(cfg.gpu_ids)]
        gpu_str = ",".join(str(g) for g in cfg.gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logger.info("CUDA_VISIBLE_DEVICES set to: %s", gpu_str)

    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    # ---- 3. W&B initialisation ----
    wandb_run_id = None
    wandb_resume = None
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from = getattr(cfg, "resume_from", None)
    if resume_from is not None:
        # Recover W&B run ID for seamless continuation.
        meta = load_run_metadata(cfg.output_dir)
        if meta and meta.get("wandb_run_id"):
            wandb_run_id = meta["wandb_run_id"]
            wandb_resume = "must"
            logger.info("Resuming W&B run: %s", wandb_run_id)

    os.environ["WANDB_PROJECT"] = getattr(cfg, "wandb_project", "vanilla-s2g")
    if getattr(cfg, "wandb_entity", None):
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    if wandb_run_id:
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = wandb_resume
    if getattr(cfg, "wandb_run_name", None):
        os.environ["WANDB_NAME"] = cfg.wandb_run_name

    # ---- 4. Load data and schema ----
    schema = load_schema(cfg.schema_file)
    logger.info("Loaded schema with %d relation types.", len(schema))

    train_dataset = S2GDataset(Path(cfg.data_dir) / "train.jsonl", seed=cfg.seed)
    val_dataset = S2GDataset(
        Path(cfg.data_dir) / "val.jsonl",
        subset_fraction=getattr(cfg, "val_percent_check", None),
        seed=cfg.seed,
    )
    logger.info("Train: %d instances, Val: %d instances", len(train_dataset), len(val_dataset))

    # ---- 5. Initialise model and tokeniser ----
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    num_added = add_special_tokens_to_tokenizer(tokenizer, model)
    logger.info("Model loaded: %s (%d special tokens added)", cfg.model_name, num_added)

    # ---- 6. Create collators (train + eval) ----
    train_collator_config = {
        "max_source_length": cfg.max_source_length,
        "max_target_length": cfg.max_target_length,
        "max_steps": cfg.max_steps,
        "positive_rate": cfg.positive_rate,
        "negative_rate": cfg.negative_rate,
        "negative_max_start": cfg.negative_max_start,
        "negative_max_end": cfg.negative_max_end,
        "random_prompt": cfg.random_prompt,
        "random_sel": cfg.random_sel,
    }
    train_collator = S2GCollator(tokenizer, schema, train_collator_config)

    # Eval collator: all gold positives included, negatives capped at k(t)
    # (same schedule as training).  Step counter is shared with the train
    # collator so that the eval collator uses the same k(t) the model has
    # been trained to handle at each validation check.
    eval_collator_config = {
        "max_source_length": cfg.max_source_length,
        "max_target_length": cfg.max_target_length,
        "max_steps": cfg.max_steps,
        "positive_rate": 1.0,
        "negative_rate": 1.0,
        "negative_max_start": cfg.negative_max_start,
        "negative_max_end": cfg.negative_max_end,
        "random_prompt": False,
        "random_sel": False,
    }
    eval_collator = S2GCollator(tokenizer, schema, eval_collator_config)
    eval_collator.share_step_with(train_collator)

    # ---- 7. Set up callbacks ----
    callbacks = []

    # Step tracking for dynamic negative cap schedule.
    step_tracker = StepTrackingCallback(train_collator)
    callbacks.append(step_tracker)

    # Early stopping on validation F1.
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
        )
    )

    # Periodic safety-net checkpoints.
    checkpoint_every = getattr(cfg, "checkpoint_every_n_steps", 5000)
    wandb_run_id_for_meta = wandb_run_id  # May be set later by W&B init.
    periodic_ckpt = PeriodicCheckpointCallback(
        output_dir=cfg.output_dir,
        every_n_steps=checkpoint_every,
        wandb_run_id=wandb_run_id_for_meta,
    )
    callbacks.append(periodic_ckpt)

    # Sample generation table for W&B.
    sample_size = min(8, len(val_dataset))
    sample_batch = [val_dataset[i] for i in range(sample_size)]
    sample_interval = getattr(cfg, "sample_generation_interval", 50_000)
    gen_samples_cb = GenerateTextSamplesCallback(
        tokenizer=tokenizer,
        sample_batch=sample_batch,
        collator=eval_collator,
        interval=sample_interval,
        eval_beams=cfg.eval_beams,
        max_target_length=cfg.max_target_length,
    )
    callbacks.append(gen_samples_cb)

    # ---- 8. Configure TrainingArguments ----
    num_gpus = len(cfg.gpu_ids) if (hasattr(cfg, "gpu_ids") and cfg.gpu_ids) else max(torch.cuda.device_count(), 1)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,

        # Training loop.
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.train_batch_size,
        gradient_accumulation_steps=cfg.gradient_acc_steps,
        max_grad_norm=cfg.gradient_clip_value,
        fp16=(cfg.precision == 16),
        bf16=(cfg.precision == "bf16"),
        dataloader_num_workers=cfg.num_workers,
        seed=cfg.seed,
        data_seed=cfg.seed,

        # Optimiser (Trainer creates AdamW internally).
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,

        # Scheduler: handled by S2GTrainer.create_scheduler override.
        # Set to "constant" to prevent Trainer from creating its own.
        lr_scheduler_type="constant",

        # Evaluation.
        eval_strategy="steps",
        eval_steps=cfg.val_check_interval,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        generation_num_beams=cfg.eval_beams,

        # Checkpointing.
        save_strategy="steps",
        save_steps=cfg.val_check_interval,
        save_total_limit=cfg.save_top_k + 1,
        load_best_model_at_end=True,
        metric_for_best_model="boundary_f1",
        greater_is_better=True,

        # Logging.
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        run_name=getattr(cfg, "wandb_run_name", None),

        # Misc.
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # ---- 9. Create Trainer ----
    trainer = S2GTrainer(
        warmup_steps=cfg.warmup_steps,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    # Override the eval collator.  The Trainer uses ``data_collator`` for
    # both train and eval by default.  We swap in the eval collator for
    # the evaluation DataLoader by overriding ``get_eval_dataloader``.
    _original_get_eval_dl = trainer.get_eval_dataloader

    def _patched_get_eval_dataloader(eval_dataset=None):
        # Temporarily swap collator, build dataloader, restore.
        original_collator = trainer.data_collator
        trainer.data_collator = eval_collator
        dl = _original_get_eval_dl(eval_dataset)
        trainer.data_collator = original_collator
        return dl

    trainer.get_eval_dataloader = _patched_get_eval_dataloader

    # ---- 10. Update W&B run ID after init (for metadata persistence) ----
    try:
        import wandb
        if wandb.run is not None:
            periodic_ckpt.wandb_run_id = wandb.run.id
    except ImportError:
        pass

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
    logger.info("Running final evaluation on validation set...")
    val_metrics = trainer.evaluate()
    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info("Validation metrics saved to %s", metrics_path)
    logger.info("Final val metrics: %s", val_metrics)


if __name__ == "__main__":
    main()