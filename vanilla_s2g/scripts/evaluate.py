"""
Evaluation script for the Vanilla S2G model.

Loads a trained checkpoint, runs (optionally constrained) generation on
a specified data split, parses the SEL output, computes corpus-level
metrics, and writes the output files specified in Section 5.2 of the
documentation.

The Schema-Sensitive-Input (SSI) prompt is capped to a configurable
number of relation types, mirroring the budget-mode logic of
:class:`~vanilla_s2g.data.S2GCollator` used during fine-tuning (and the
tail end of pre-training).  All gold positives are always included; the
remaining budget is filled with negatives sampled uniformly from the
schema's negative pool.  Setting ``ssi.max_types_in_prompt: null``
restores the legacy behaviour of prompting with the full schema.

Usage::

    # Standard run with the YAML defaults
    python -m vanilla_s2g.scripts.evaluate --config configs/evaluate.yaml

    # Override fields via dotlist
    python -m vanilla_s2g.scripts.evaluate --config configs/evaluate.yaml \\
        model.pretrained_checkpoint=outputs/pretrain/best_model \\
        evaluation.split=val ssi.max_types_in_prompt=15

Output files (written to ``cfg.data.output_dir``)::

    {split}_out.jsonl       — Generated SEL strings.
    {split}_preds.jsonl     — Parsed entities, relations, and rejected types.
    {split}_results.jsonl   — Original text with gold and predicted structures.
    {split}_metrics.json    — All evaluation metrics.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from vanilla_s2g.data import S2GDataset
from vanilla_s2g.evaluation import compute_metrics
from vanilla_s2g.linearisation import (
    add_special_tokens_to_tokenizer,
    build_encoder_input,
    extract_triplets,
    parse_sel,
)
from vanilla_s2g.model import build_constraint_processor
from vanilla_s2g.scripts.config_utils import load_config, load_schema

logger = logging.getLogger(__name__)


# ===================================================================== #
#                       SSI BUDGET-MODE SAMPLING                         #
# ===================================================================== #


def _sample_capped_ssi_types(
    schema: List[str],
    instance_types: List[str],
    max_types_in_prompt: int,
    rng: random.Random,
) -> List[str]:
    """Sample SSI types so that the total prompt is capped.

    Mirrors the *budget-mode* branch of
    :meth:`vanilla_s2g.data.S2GCollator._sample_types`:

    - All gold positives are always included (``positive_rate = 1.0``).
    - Negatives are drawn uniformly from ``schema − instance_types`` to
      fill the remaining budget ``max_types_in_prompt − len(positives)``.
    - If gold positives alone already exceed the budget, no negatives
      are added (positives are *not* truncated, matching the collator).

    A scoped ``random.Random`` instance is used so that sampling is
    deterministic and isolated from the global random state.

    Args:
        schema:               Full list of relation-type strings.
        instance_types:       Gold relation types present in the instance.
        max_types_in_prompt:  Hard cap on positives + negatives in the SSI.
        rng:                  Scoped RNG for negative sampling.

    Returns:
        Concatenated list of positive then negative types, ready for
        :func:`build_encoder_input`.
    """
    instance_set = set(instance_types)
    negative_pool = [t for t in schema if t not in instance_set]

    neg_budget = max(0, max_types_in_prompt - len(instance_types))
    n_neg = min(neg_budget, len(negative_pool))
    sampled_negatives = rng.sample(negative_pool, n_neg) if n_neg > 0 else []

    return list(instance_types) + sampled_negatives


# ===================================================================== #
#                              EVALUATE                                  #
# ===================================================================== #


def evaluate(
    model: Any,
    tokenizer: Any,
    dataset: S2GDataset,
    schema: List[str],
    output_dir: Path,
    split: str = "test",
    constraint_decoding: bool = True,
    eval_beams: int = 3,
    max_source_length: int = 300,
    max_target_length: int = 150,
    batch_size: int = 64,
    mode: str = "boundary",
    max_types_in_prompt: Optional[int] = None,
    random_prompt: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    """Run full evaluation and write output files.

    Args:
        model:               The seq2seq model.
        tokenizer:           HuggingFace tokeniser with S2G tokens.
        dataset:             The evaluation dataset.
        schema:              List of all relation-type strings.
        output_dir:          Directory for output files.
        split:               Split name (``"val"`` or ``"test"``).
        constraint_decoding: Whether to activate FSM constraints.
        eval_beams:          Number of beams for generation.
        max_source_length:   Encoder max length.
        max_target_length:   Decoder max length.
        batch_size:          Inference batch size.
        mode:                ``"boundary"`` or ``"strict"``.
        max_types_in_prompt: If set, cap the SSI to this many total types
                             per instance (positives + sampled negatives).
                             ``None`` reproduces the legacy full-schema
                             behaviour.
        random_prompt:       Shuffle SSI type order (otherwise sorted
                             alphabetically by :func:`build_ssi_prefix`).
        seed:                Seed for the deterministic negative-sampling
                             RNG.

    Returns:
        Dictionary of evaluation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    # Scoped RNG: independent of the global random state, so the
    # negative-sample sequence is reproducible across runs given the
    # same seed regardless of what other library code may have done.
    rng = random.Random(seed)

    if max_types_in_prompt is not None:
        logger.info(
            "SSI capped at %d types per instance (gold positives + sampled negatives).",
            max_types_in_prompt,
        )
    else:
        logger.info(
            "SSI cap disabled — prompting with the full schema (%d types).",
            len(schema),
        )

    # Prepare output file handles.
    out_path = output_dir / f"{split}_out.jsonl"
    preds_path = output_dir / f"{split}_preds.jsonl"
    results_path = output_dir / f"{split}_results.jsonl"

    # Accumulate for corpus-level metrics.
    all_pred_triplets: List[list] = []
    all_gold_triplets: List[list] = []
    all_pred_entities: List[list] = []
    all_gold_entities: List[list] = []

    f_out = open(out_path, "w", encoding="utf-8")
    f_preds = open(preds_path, "w", encoding="utf-8")
    f_results = open(results_path, "w", encoding="utf-8")

    try:
        # Process in batches.
        num_instances = len(dataset)
        for start_idx in tqdm(range(0, num_instances, batch_size), desc=f"Evaluating {split}"):
            end_idx = min(start_idx + batch_size, num_instances)
            batch_instances = [dataset[i] for i in range(start_idx, end_idx)]

            # Build encoder inputs.  When the cap is enabled, each
            # instance's SSI contains its gold positives plus a separate
            # uniform sample of negatives drawn from the schema's
            # negative pool; otherwise the full schema is used.  In
            # either case the constraint decoder will later read the
            # prompted types straight out of ``source_ids`` and restrict
            # decoding accordingly, so no separate bookkeeping of
            # per-instance label lists is needed here.
            if max_types_in_prompt is not None:
                encoder_texts = [
                    build_encoder_input(
                        _sample_capped_ssi_types(
                            schema=schema,
                            instance_types=inst["types"],
                            max_types_in_prompt=max_types_in_prompt,
                            rng=rng,
                        ),
                        inst["text"],
                        random_prompt=random_prompt,
                    )
                    for inst in batch_instances
                ]
            else:
                encoder_texts = [
                    build_encoder_input(schema, inst["text"], random_prompt=random_prompt)
                    for inst in batch_instances
                ]

            encoded = tokenizer(
                encoder_texts,
                max_length=max_source_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Generation kwargs.
            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "num_beams": eval_beams,
                "max_length": max_target_length,
                "length_penalty": 0.0,
                "no_repeat_ngram_size": 0,
                "early_stopping": False,
            }

            # Constraint decoding.  The processor reads the prompted
            # relation types directly out of ``input_ids`` (one trie
            # pair per batch item), so the decoder is restricted to
            # exactly the labels that appear in each instance's SSI.
            # The source-copy constraint also uses ``input_ids``, so
            # both halves of the FSM stay consistent with whatever the
            # encoder saw.
            if constraint_decoding:
                processor = build_constraint_processor(
                    tokenizer=tokenizer,
                    source_ids=input_ids,
                    num_beams=eval_beams,
                )
                gen_kwargs["logits_processor"] = [processor]

            with torch.no_grad():
                generated_ids = model.generate(**gen_kwargs)

            # Decode, parse, and write.
            for i, inst in enumerate(batch_instances):
                # Predicted SEL.
                pred_sel = tokenizer.decode(
                    generated_ids[i], skip_special_tokens=False
                )
                pred_sel = _clean_decoded(pred_sel, tokenizer)
                pred_ents, pred_rejected = parse_sel(pred_sel)
                pred_triplets = extract_triplets(pred_ents)

                # Gold SEL.
                gold_ents, gold_rejected = parse_sel(inst["sel"])
                gold_triplets = extract_triplets(gold_ents)

                # Accumulate.
                all_pred_triplets.append(pred_triplets)
                all_gold_triplets.append(gold_triplets)
                all_pred_entities.append([e["text"] for e in pred_ents])
                all_gold_entities.append([e["text"] for e in gold_ents])

                # Write output SEL.
                f_out.write(json.dumps({"sel": pred_sel}, ensure_ascii=False) + "\n")

                # Write parsed predictions.
                f_preds.write(json.dumps({
                    "entities": pred_ents,
                    "relations": [
                        {"head": e["text"], "type": r["type"], "tail": r["tail"]}
                        for e in pred_ents for r in e["relations"]
                    ],
                    "rejected": pred_rejected,
                }, ensure_ascii=False) + "\n")

                # Write full results.
                f_results.write(json.dumps({
                    "text": inst["text"],
                    "gold_sel": inst["sel"],
                    "pred_sel": pred_sel,
                    "gold_entities": [{"text": e["text"], "relations": e["relations"]}
                                      for e in gold_ents],
                    "pred_entities": pred_ents,
                    "gold_triplets": [list(t) for t in gold_triplets],
                    "pred_triplets": [list(t) for t in pred_triplets],
                }, ensure_ascii=False) + "\n")

    finally:
        f_out.close()
        f_preds.close()
        f_results.close()

    # Compute corpus-level metrics.
    metrics = compute_metrics(
        all_pred_triplets, all_gold_triplets,
        all_pred_entities, all_gold_entities,
        mode=mode,
    )

    # Write metrics.
    metrics_path = output_dir / f"{split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("%s metrics: %s", split, json.dumps(metrics, indent=2))
    logger.info("Output files written to %s", output_dir)

    return metrics


def _clean_decoded(text: str, tokenizer) -> str:
    """Strip decoder artefacts from a decoded string."""
    for tok in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]:
        if tok:
            text = text.replace(tok, "")
    return " ".join(text.split())


# ===================================================================== #
#                                MAIN                                    #
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
        gpu_str = ",".join(str(g) for g in cfg.hardware.gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logger.info("CUDA_VISIBLE_DEVICES set to: %s", gpu_str)

    set_seed(cfg.train.seed)
    logger.info("Random seed set to %d", cfg.train.seed)

    # ---- 3. Resolve checkpoint path ----
    checkpoint = cfg.model.pretrained_checkpoint
    if checkpoint is None:
        raise ValueError(
            "model.pretrained_checkpoint is required for evaluation. "
            "Set it in the YAML or via "
            "'model.pretrained_checkpoint=<path>' on the CLI."
        )

    # ---- 4. Load model and tokeniser ----
    logger.info("Loading model from %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Ensure special tokens are registered (they should already be in
    # the saved tokenizer, but this is a safety net).
    add_special_tokens_to_tokenizer(tokenizer, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model loaded on %s", device)

    # ---- 5. Load data and schema ----
    schema = load_schema(cfg.data.schema_file)
    logger.info("Loaded schema with %d relation types.", len(schema))

    split = cfg.evaluation.split
    if split not in ("val", "test"):
        raise ValueError(
            f"evaluation.split must be 'val' or 'test', got '{split}'."
        )
    split_file = {"val": "val.jsonl", "test": "test.jsonl"}[split]
    dataset = S2GDataset(
        Path(cfg.data.data_dir) / split_file,
        seed=cfg.train.seed,
    )
    logger.info("%s set: %d instances", split, len(dataset))

    # ---- 6. Run evaluation ----
    evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        schema=schema,
        output_dir=Path(cfg.data.output_dir),
        split=split,
        constraint_decoding=cfg.generation.constraint_decoding,
        eval_beams=cfg.generation.num_beams,
        max_source_length=cfg.tokenization.max_source_length,
        max_target_length=cfg.tokenization.max_target_length,
        batch_size=cfg.validation.batch_size,
        mode=cfg.evaluation.mode,
        max_types_in_prompt=cfg.ssi.max_types_in_prompt,
        random_prompt=cfg.ssi.random_prompt,
        seed=cfg.train.seed,
    )


if __name__ == "__main__":
    main()