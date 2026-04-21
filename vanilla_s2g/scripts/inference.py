"""
Inference script for the Vanilla S2G model.

Provides a simple interface for extracting knowledge-graph triplets from
arbitrary text using a trained S2G model.  Supports both interactive
mode (type sentences at the prompt) and batch mode (read from a file).

Usage::

    # Interactive mode
    python -m vanilla_s2g.scripts.inference \\
        --checkpoint outputs/pretrain/best_model \\
        --schema_file data/rebel/relation.schema

    # Batch mode (one sentence per line)
    python -m vanilla_s2g.scripts.inference \\
        --checkpoint outputs/pretrain/best_model \\
        --schema_file data/rebel/relation.schema \\
        --input_file sentences.txt \\
        --output_file triplets.jsonl

    # With specific relation types only
    python -m vanilla_s2g.scripts.inference \\
        --checkpoint outputs/pretrain/best_model \\
        --schema_file data/rebel/relation.schema \\
        --relation_types "place of birth" "president of"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from vanilla_s2g.linearisation import (
    add_special_tokens_to_tokenizer,
    build_encoder_input,
    extract_triplets,
    parse_sel,
)
from vanilla_s2g.model import build_constraint_processor
from vanilla_s2g.scripts.config_utils import load_schema

logger = logging.getLogger(__name__)


def extract(
    text: str,
    model: Any,
    tokenizer: Any,
    schema: List[str],
    relation_types: Optional[List[str]] = None,
    constraint_decoding: bool = True,
    num_beams: int = 3,
    max_source_length: int = 300,
    max_target_length: int = 150,
) -> Dict[str, Any]:
    """Extract triplets from a single sentence.

    Args:
        text:                Raw input sentence.
        model:               The seq2seq model (on device).
        tokenizer:           HuggingFace tokeniser with S2G tokens.
        schema:              Full list of relation-type strings.
        relation_types:      Subset of types to query for (``None`` = full schema).
        constraint_decoding: Whether to activate FSM constraints.
        num_beams:           Number of beams for generation.
        max_source_length:   Encoder max length.
        max_target_length:   Decoder max length.

    Returns:
        Dictionary with keys: ``text``, ``sel``, ``entities``,
        ``triplets``, ``rejected``.
    """
    device = next(model.parameters()).device
    types_in_scope = relation_types if relation_types else schema

    # Build encoder input.
    encoder_input = build_encoder_input(types_in_scope, text, random_prompt=False)

    encoded = tokenizer(
        [encoder_input],
        max_length=max_source_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "num_beams": num_beams,
        "max_length": max_target_length,
        "length_penalty": 0.0,
        "no_repeat_ngram_size": 0,
        "early_stopping": False,
    }

    if constraint_decoding:
        processor = build_constraint_processor(
            tokenizer=tokenizer,
            source_ids=input_ids,
            relation_types=types_in_scope,
            num_beams=num_beams,
        )
        gen_kwargs["logits_processor"] = [processor]

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)

    # Decode and parse.
    raw_sel = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    for tok in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]:
        if tok:
            raw_sel = raw_sel.replace(tok, "")
    raw_sel = " ".join(raw_sel.split())

    entities, rejected = parse_sel(raw_sel)
    triplets = extract_triplets(entities)

    return {
        "text": text,
        "sel": raw_sel,
        "entities": entities,
        "triplets": [{"head": h, "type": r, "tail": t} for h, r, t in triplets],
        "rejected": rejected,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract knowledge-graph triplets from text."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--schema_file", type=str, required=True,
                        help="Path to the relation.schema file.")
    parser.add_argument("--input_file", type=str, default=None,
                        help="File with one sentence per line (batch mode).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL file (batch mode).")
    parser.add_argument("--relation_types", type=str, nargs="*", default=None,
                        help="Specific relation types to query (default: full schema).")
    parser.add_argument("--constraint_decoding", type=str, default="true")
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_source_length", type=int, default=300)
    parser.add_argument("--max_target_length", type=int, default=150)
    args = parser.parse_args()

    # Load model.
    logger.info("Loading model from %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    add_special_tokens_to_tokenizer(tokenizer, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model loaded on %s", device)

    schema = load_schema(args.schema_file)
    constraint_flag = args.constraint_decoding.lower() in ("true", "1", "yes")

    common_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "schema": schema,
        "relation_types": args.relation_types,
        "constraint_decoding": constraint_flag,
        "num_beams": args.num_beams,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }

    if args.input_file:
        # ---- Batch mode ----
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

        with open(input_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        logger.info("Processing %d sentences...", len(sentences))
        output_handle = (
            open(args.output_file, "w", encoding="utf-8")
            if args.output_file
            else sys.stdout
        )

        try:
            for sentence in sentences:
                result = extract(text=sentence, **common_kwargs)
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        finally:
            if args.output_file:
                output_handle.close()

        logger.info("Done. Output: %s", args.output_file or "stdout")

    else:
        # ---- Interactive mode ----
        print("\n=== S2G Interactive Inference ===")
        print(f"Schema: {len(schema)} relation types loaded")
        if args.relation_types:
            print(f"Querying: {args.relation_types}")
        print('Type a sentence and press Enter. Type "quit" to exit.\n')

        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not text or text.lower() in ("quit", "exit", "q"):
                break

            result = extract(text=text, **common_kwargs)

            if result["triplets"]:
                print(f"\n  Triplets ({len(result['triplets'])}):")
                for t in result["triplets"]:
                    print(f"    ({t['head']}, {t['type']}, {t['tail']})")
            else:
                print("\n  No triplets extracted.")

            if result["rejected"]:
                print(f"  Rejected types: {len(result['rejected'])}")
            print()


if __name__ == "__main__":
    main()