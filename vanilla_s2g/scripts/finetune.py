"""
Fine-tuning script for the Vanilla S2G model.

Loads a pre-trained checkpoint, registers special tokens and optional typed SEL
tokens, and fine-tunes the model on relation extraction datasets (CoNLL 2004, NYT, SciERC)
using static SSI construction and early stopping on averaged strict + boundary F1.

Usage::

    python -m vanilla_s2g.scripts.finetune --config configs/conll04.yaml
    python -m vanilla_s2g.scripts.finetune --config configs/nyt.yaml
    python -m vanilla_s2g.scripts.finetune --config configs/scierc.yaml
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
    extract_quintuples,
    parse_sel,
)
from vanilla_s2g.scripts.config_utils import load_config, load_schema

logger = logging.getLogger(__name__)


# ===================================================================== #
#                     CUSTOM TRAINER SUBCLASS                            #
# ===================================================================== #


class S2GTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with inverse square root scheduler support."""

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
        eval_dataloader = self.get_eval_dataloader(self.train_eval_dataset)
        eval_loop = self.evaluation_loop

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


def make_compute_metrics(tokenizer, cfg):
    """Build a compute_metrics function for the Seq2SeqTrainer."""
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions == -100, pad_id, predictions)
        label_ids = np.where(label_ids == -100, pad_id, label_ids)

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=False)
        gold_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=False)

        all_pred_items = []
        all_gold_items = []
        all_pred_entities = []
        all_gold_entities = []

        for pred_text, gold_text in zip(pred_texts, gold_texts):
            pred_text = _clean_decoded(pred_text, tokenizer)
            gold_text = _clean_decoded(gold_text, tokenizer)

            pred_ents, _ = parse_sel(pred_text)
            gold_ents, _ = parse_sel(gold_text)

            if cfg.typed_sel.enabled:
                all_pred_items.append(extract_quintuples(pred_ents))
                all_gold_items.append(extract_quintuples(gold_ents))
                all_pred_entities.append([(e["text"], e.get("type", "")) for e in pred_ents])
                all_gold_entities.append([(e["text"], e.get("type", "")) for e in gold_ents])
            else:
                all_pred_items.append(extract_triplets(pred_ents))
                all_gold_items.append(extract_triplets(gold_ents))
                all_pred_entities.append([e["text"] for e in pred_ents])
                all_gold_entities.append([e["text"] for e in gold_ents])

        eval_mode = "strict" if cfg.typed_sel.enabled else "boundary"
        return eval_compute_metrics(
            all_pred_items, all_gold_items,
            all_pred_entities, all_gold_entities,
            mode=eval_mode,
            typed_ner=cfg.typed_sel.enabled,
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
    if cfg.hardware.gpu_ids is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            logger.warning(
                "hardware.gpu_ids is set but WORLD_SIZE > 1: GPU assignment is "
                "managed by torchrun in distributed mode; gpu_ids is ignored."
            )
        else:
            gpu_str = ",".join(str(g) for g in cfg.hardware.gpu_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            logger.info("CUDA_VISIBLE_DEVICES set to: %s", gpu_str)

    set_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    logger.info("Random seed set to %d", cfg.train.seed)

    # ---- 3. W&B initialisation ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    wandb_run_id: Optional[str] = None
    wandb_resume: Optional[str] = None
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from = cfg.checkpoint.resume_from
    if resume_from is not None:
        meta = load_run_metadata(cfg.data.output_dir)
        if meta and meta.get("wandb_run_id"):
            wandb_run_id = meta["wandb_run_id"]
            wandb_resume = "must"
            logger.info("Resuming W&B run: %s", wandb_run_id)

    if local_rank == 0:                           
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            id=wandb_run_id,
            resume=wandb_resume,
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
    checkpoint_to_load = cfg.model.pretrained_checkpoint or cfg.model.name
    logger.info("Loading tokenizer and model from: %s", checkpoint_to_load)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_to_load)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_to_load)

    # Add standard S2G special tokens
    num_added = add_special_tokens_to_tokenizer(tokenizer, model)
    logger.info("Registered standard S2G tokens (%d added)", num_added)

    # Add typed SEL tokens (if enabled) and copy embeddings from <ent>
    if cfg.typed_sel.enabled and cfg.typed_sel.entity_types:
        typed_tokens = [f"<{t.replace(' ', '_')}>" for t in cfg.typed_sel.entity_types]
        num_added_typed = tokenizer.add_special_tokens({"additional_special_tokens": typed_tokens})
        if num_added_typed > 0:
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            ent_id = tokenizer.convert_tokens_to_ids("<ent>")
            for t in typed_tokens:
                t_id = tokenizer.convert_tokens_to_ids(t)
                with torch.no_grad():
                    input_embeddings.weight[t_id].copy_(input_embeddings.weight[ent_id])
                    if output_embeddings is not None:
                        output_embeddings.weight[t_id].copy_(output_embeddings.weight[ent_id])
            logger.info("Registered %d typed SEL tokens and initialized from <ent>", num_added_typed)

    # ---- 6. Create collators (train + eval) ----
    train_collator_config = {
        "mode": "static",
        "max_source_length": cfg.tokenization.max_source_length,
        "max_target_length": cfg.tokenization.max_target_length,
        "random_prompt": cfg.ssi.random_prompt,
        "random_sel": cfg.ssi.random_sel,
        "typed_sel_enabled": cfg.typed_sel.enabled,
    }
    train_collator = S2GCollator(tokenizer, schema, train_collator_config)

    eval_collator_config = {
        "mode": "static",
        "max_source_length": cfg.tokenization.max_source_length,
        "max_target_length": cfg.tokenization.max_target_length,
        "random_prompt": cfg.ssi.random_prompt,
        "random_sel": cfg.ssi.random_sel,
        "typed_sel_enabled": cfg.typed_sel.enabled,
    }
    eval_collator = S2GCollator(tokenizer, schema, eval_collator_config)

    # ---- 7. Set up callbacks ----
    callbacks = []

    # Step tracking (collator uses static mode, but tracker kept for alignment)
    step_tracker = StepTrackingCallback(train_collator)
    callbacks.append(step_tracker)

    # Early stopping on validation F1
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=cfg.validation.early_stopping_patience,
        )
    )

    # Periodic safety-net checkpoints
    periodic_ckpt = PeriodicCheckpointCallback(
        output_dir=cfg.data.output_dir,
        every_n_steps=cfg.checkpoint.every_n_steps,
        wandb_run_id=wandb.run.id if wandb.run is not None else None,
    )
    callbacks.append(periodic_ckpt)

    # Sample generation table for W&B
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
    metric_for_best = cfg.checkpoint.metric
    if metric_for_best.startswith("val_"):
        metric_for_best = metric_for_best[4:]  # e.g., "avg_f1"

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.data.output_dir,

        # Training loop
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_acc_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.train.gradient_checkpointing else None,
        max_grad_norm=cfg.train.gradient_clip_value,
        fp16=(cfg.train.precision == "16"),
        bf16=(cfg.train.precision == "bf16"),
        dataloader_num_workers=cfg.hardware.num_workers,
        dataloader_persistent_workers=cfg.hardware.persistent_workers,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,

        # Optimiser
        optim=cfg.optimizer.optim,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        adam_epsilon=cfg.optimizer.adam_epsilon,

        # Scheduler
        warmup_steps=cfg.scheduler.warmup_steps,
        lr_scheduler_type=cfg.scheduler.type if cfg.scheduler.type != "inverse_sqrt" else "constant",

        # Evaluation
        eval_strategy="steps",
        eval_steps=cfg.validation.check_interval,
        per_device_eval_batch_size=cfg.validation.batch_size,
        predict_with_generate=True,
        generation_max_length=cfg.tokenization.max_target_length,
        generation_num_beams=cfg.generation.num_beams,

        # Checkpointing
        save_strategy="steps",
        save_steps=cfg.validation.check_interval,
        save_total_limit=cfg.checkpoint.save_top_k + 1,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,

        # Logging
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        run_name=cfg.wandb.run_name,

        # Misc
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
        compute_metrics=make_compute_metrics(tokenizer, cfg),
        callbacks=callbacks,
    )

    # ---- 10. Train ----
    logger.info("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Fine-tuning complete.")

    # ---- 11. Save best model ----
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Best model saved to %s", best_dir)

    # ---- 12. Final evaluation on validation set ----
    logger.info("Loading full validation set for final evaluation...")
    full_val_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "val.jsonl",
        seed=cfg.train.seed, 
    )
    logger.info("Full Val: %d instances", len(full_val_dataset))

    logger.info("Running final evaluation on full validation set...")
    val_metrics = trainer.evaluate(eval_dataset=full_val_dataset)
    
    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info("Validation metrics saved to %s", metrics_path)
    logger.info("Final val metrics: %s", val_metrics)


if __name__ == "__main__":
    main()