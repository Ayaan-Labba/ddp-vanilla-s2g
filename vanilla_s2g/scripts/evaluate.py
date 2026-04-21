"""
Evaluation script for the Vanilla S2G model.

Loads a trained checkpoint, runs constrained generation on a specified
data split, parses the SEL output, computes all metrics, and writes
the output files specified in Section 5.2 of the documentation.

Usage::

    python -m vanilla_s2g.scripts.evaluate \\
        --checkpoint outputs/pretrain/best_model \\
        --data_dir data/rebel \\
        --schema_file data/rebel/relation.schema \\
        --split test \\
        --output_dir outputs/pretrain/eval \\
        --constraint_decoding true \\
        --eval_beams 3

Output files::

    {split}_out.jsonl       — Generated SEL strings.
    {split}_preds.jsonl     — Parsed entities, relations, and rejected types.
    {split}_results.jsonl   — Original text with gold and predicted structures.
    {split}_metrics.json    — All evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from vanilla_s2g.data import S2GCollator, S2GDataset
from vanilla_s2g.evaluation import compute_metrics
from vanilla_s2g.linearisation import (
    add_special_tokens_to_tokenizer,
    build_encoder_input,
    extract_triplets,
    get_token_ids,
    parse_sel,
)
from vanilla_s2g.model import build_constraint_processor
from vanilla_s2g.scripts.config_utils import load_schema

logger = logging.getLogger(__name__)


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
    batch_size: int = 8,
    mode: str = "boundary",
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

    Returns:
        Dictionary of evaluation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

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

            # Build encoder inputs with full schema.
            encoder_texts = [
                build_encoder_input(schema, inst["text"], random_prompt=False)
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

            # Constraint decoding.
            if constraint_decoding:
                processor = build_constraint_processor(
                    tokenizer=tokenizer,
                    source_ids=input_ids,
                    relation_types=schema,
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
#                            MAIN                                        #
# ===================================================================== #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained S2G model.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the JSONL data files.")
    parser.add_argument("--schema_file", type=str, required=True,
                        help="Path to the relation.schema file.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Which split to evaluate on.")
    parser.add_argument("--output_dir", type=str, default="outputs/eval",
                        help="Directory for output files.")
    parser.add_argument("--constraint_decoding", type=str, default="true",
                        help="Enable FSM constraint decoding (true/false).")
    parser.add_argument("--eval_beams", type=int, default=3)
    parser.add_argument("--max_source_length", type=int, default=300)
    parser.add_argument("--max_target_length", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mode", type=str, default="boundary",
                        choices=["boundary", "strict"],
                        help="Metric mode.")
    args = parser.parse_args()

    # Load model and tokeniser.
    logger.info("Loading model from %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)

    # Ensure special tokens are registered (they should already be in the
    # saved tokenizer, but this is a safety net).
    add_special_tokens_to_tokenizer(tokenizer, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model loaded on %s", device)

    # Load data and schema.
    schema = load_schema(args.schema_file)
    split_file = {"val": "val.jsonl", "test": "test.jsonl"}[args.split]
    dataset = S2GDataset(Path(args.data_dir) / split_file)

    # Run evaluation.
    constraint_flag = args.constraint_decoding.lower() in ("true", "1", "yes")
    evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        schema=schema,
        output_dir=Path(args.output_dir),
        split=args.split,
        constraint_decoding=constraint_flag,
        eval_beams=args.eval_beams,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        batch_size=args.batch_size,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()