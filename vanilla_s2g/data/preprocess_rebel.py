"""
Pre-processing script for the REBEL sentence-level dataset.

Downloads the REBEL dataset from HuggingFace, filters instances to those
containing only relation types from the top-K most frequent types, converts
each instance into the S2G standardised JSON format, and writes the result
as JSONL files with an accompanying relation schema file.

Usage::

    python -m vanilla_s2g.data.preprocess_rebel \\
        --output_dir data/rebel \\
        --top_k 220

The output directory will contain::

    data/rebel/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    └── relation.schema     # one relation type per line
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import nltk
from datasets import load_dataset
from tqdm import tqdm

from vanilla_s2g.linearisation import build_sel, organize_by_entity

logger = logging.getLogger(__name__)

# ===================================================================== #
#                   REBEL TRIPLET PARSING                                #
# ===================================================================== #


def parse_rebel_triplets(
    triplet_str: str,
) -> List[Tuple[str, str, str]]:
    """Parse a REBEL linearised triplet string into structured triplets.

    The REBEL format encodes triplets as::

        <triplet> head_entity <subj> tail_1 <obj> rel_1 <subj> tail_2 <obj> rel_2
        <triplet> head_entity_2 <subj> tail_3 <obj> rel_3

    A single ``<triplet>`` block can contain **multiple**
    ``<subj>...<obj>`` pairs that all share the same head entity.

    Args:
        triplet_str: Raw ``triplets`` field from the REBEL dataset.

    Returns:
        List of ``(head_text, relation_type, tail_text)`` tuples.
    """
    triplets: List[Tuple[str, str, str]] = []

    # Split by the <triplet> marker; the first element is empty or whitespace.
    segments = triplet_str.split("<triplet>")

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if "<subj>" not in segment or "<obj>" not in segment:
            logger.debug("Skipping malformed triplet segment: %s", segment)
            continue

        # Split on <subj> to separate head from the relation pairs.
        # Element 0 is the head entity; elements 1..N each contain one
        # "tail <obj> relation_type" pair.
        subj_parts = segment.split("<subj>")
        head = subj_parts[0].strip()

        if not head:
            logger.debug("Skipping segment with empty head: %s", segment)
            continue

        for subj_part in subj_parts[1:]:
            subj_part = subj_part.strip()
            if "<obj>" not in subj_part:
                logger.debug(
                    "Skipping malformed <subj> block (no <obj>): %s", subj_part
                )
                continue

            tail_rest = subj_part.split("<obj>", maxsplit=1)
            tail = tail_rest[0].strip()
            rel_type = tail_rest[1].strip()

            if tail and rel_type:
                triplets.append((head, rel_type, tail))
            else:
                logger.debug(
                    "Skipping incomplete triplet: head=%r, rel=%r, tail=%r",
                    head, rel_type, tail,
                )

    return triplets


# ===================================================================== #
#                   TOKEN-LEVEL ENTITY MATCHING                          #
# ===================================================================== #


def find_token_span(
    tokens: List[str],
    entity_text: str,
    occupied: Optional[Set[int]] = None,
) -> Optional[Tuple[int, int]]:
    """Find the first token-level span matching *entity_text*.

    Tokenises *entity_text* with NLTK and searches for an exact token-
    sequence match within *tokens*.  Falls back to case-insensitive
    matching if the exact match fails.

    Args:
        tokens:      NLTK-tokenised source text.
        entity_text: Raw entity surface form to locate.
        occupied:    Set of token indices already assigned to other
                     entities (used to prefer non-overlapping spans).

    Returns:
        ``(start, end)`` token indices (end-exclusive), or ``None``.
    """
    ent_tokens = nltk.word_tokenize(entity_text)
    n = len(ent_tokens)

    if n == 0 or n > len(tokens):
        return None

    # Pass 1: exact match, preferring non-occupied spans.
    candidates: List[Tuple[int, int]] = []
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == ent_tokens:
            span = (i, i + n)
            if occupied is None or not any(j in occupied for j in range(i, i + n)):
                return span
            candidates.append(span)

    if candidates:
        return candidates[0]

    # Pass 2: case-insensitive match.
    ent_lower = [t.lower() for t in ent_tokens]
    for i in range(len(tokens) - n + 1):
        if [t.lower() for t in tokens[i : i + n]] == ent_lower:
            return (i, i + n)

    return None


# ===================================================================== #
#                   INSTANCE CONVERSION                                  #
# ===================================================================== #


def convert_instance(
    text: str,
    raw_triplets: List[Tuple[str, str, str]],
) -> Optional[Dict]:
    """Convert a single REBEL instance into the S2G standardised format.

    Performs NLTK tokenisation, entity-offset matching, entity-block
    construction, and SEL generation.  Returns ``None`` if no valid
    entity–relation structure can be recovered (e.g. no entities could be
    matched in the tokenised text).

    Args:
        text:          Raw input sentence.
        raw_triplets:  List of ``(head, relation_type, tail)`` from REBEL.

    Returns:
        A dictionary with keys ``text``, ``tokens``, ``entities``,
        ``relations``, ``types``, ``sel`` — or ``None`` on failure.
    """
    tokens = nltk.word_tokenize(text)

    # Collect unique entity surface forms across all triplets.
    entity_texts: Set[str] = set()
    for head, _, tail in raw_triplets:
        entity_texts.add(head)
        entity_texts.add(tail)

    # Find token offsets for each unique entity.  Process longer entities
    # first to minimise false substring matches.
    occupied: Set[int] = set()
    text_to_offset: Dict[str, Tuple[int, int]] = {}

    for ent_text in sorted(entity_texts, key=len, reverse=True):
        span = find_token_span(tokens, ent_text, occupied)
        if span is not None:
            text_to_offset[ent_text] = span
            occupied.update(range(span[0], span[1]))

    if not text_to_offset:
        return None

    # Build entity dicts.
    entities: List[Dict] = []
    seen_offsets: Set[Tuple[int, int]] = set()
    for ent_text, offset in text_to_offset.items():
        offset_key = (offset[0], offset[1])
        if offset_key not in seen_offsets:
            entities.append(
                {"text": ent_text, "offset": list(offset), "type": ""}
            )
            seen_offsets.add(offset_key)

    # Build relation dicts, skipping relations with unmatched entities.
    relations: List[Dict] = []
    for head, rel_type, tail in raw_triplets:
        if head not in text_to_offset or tail not in text_to_offset:
            continue
        head_offset = text_to_offset[head]
        tail_offset = text_to_offset[tail]
        relations.append(
            {
                "head": {"text": head, "offset": list(head_offset), "type": ""},
                "tail": {"text": tail, "offset": list(tail_offset), "type": ""},
                "type": rel_type,
            }
        )

    if not relations:
        return None

    # Derive the unique relation types present in this instance.
    types = sorted(set(r["type"] for r in relations))

    # Build the SEL (without rejected types — those are added dynamically
    # by the collator during training).
    entity_blocks = organize_by_entity(entities, relations)
    sel = build_sel(entity_blocks, rejected_types=[])

    return {
        "text": text,
        "tokens": tokens,
        "entities": entities,
        "relations": relations,
        "types": types,
        "sel": sel,
    }


# ===================================================================== #
#                   DATASET-LEVEL PROCESSING                             #
# ===================================================================== #


def count_relation_types(
    dataset,
    split: str = "train",
) -> Counter:
    """Count relation-type frequencies across all instances in a split.

    Args:
        dataset: HuggingFace dataset dict.
        split:   Which split to count (default: ``"train"``).

    Returns:
        :class:`Counter` mapping relation-type strings to counts.
    """
    counter: Counter = Counter()

    for instance in tqdm(dataset[split], desc=f"Counting types ({split})"):
        triplets = parse_rebel_triplets(instance["triplets"])
        for _, rel_type, _ in triplets:
            counter[rel_type] += 1

    return counter


def process_split(
    dataset,
    split: str,
    allowed_types: Set[str],
    output_path: Path,
) -> int:
    """Convert and write one dataset split to JSONL.

    An instance is **included** only if *every* relation type it contains
    is in *allowed_types*.  This mirrors the REBEL filtering strategy.

    Args:
        dataset:       HuggingFace dataset dict.
        split:         Split name (``"train"``, ``"validation"``, ``"test"``).
        allowed_types: Set of relation-type strings to retain.
        output_path:   Path to the output JSONL file.

    Returns:
        Number of instances written.
    """
    written = 0
    skipped_type = 0
    skipped_convert = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for instance in tqdm(dataset[split], desc=f"Processing {split}"):
            raw_triplets = parse_rebel_triplets(instance["triplets"])
            if not raw_triplets:
                skipped_convert += 1
                continue

            # Keep only if ALL relation types are in the allowed set.
            instance_types = set(rt for _, rt, _ in raw_triplets)
            if not instance_types.issubset(allowed_types):
                skipped_type += 1
                continue

            converted = convert_instance(instance["context"], raw_triplets)
            if converted is None:
                skipped_convert += 1
                continue

            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        "%s: wrote %d instances (skipped %d type-filtered, %d conversion failures)",
        split, written, skipped_type, skipped_convert,
    )
    return written


# ===================================================================== #
#                             MAIN                                       #
# ===================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process the REBEL dataset for S2G pre-training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/rebel",
        help="Directory for output JSONL and schema files.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=220,
        help="Number of most-frequent relation types to retain.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Babelscape/rebel-dataset",
        help="HuggingFace dataset identifier.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Ensure NLTK tokeniser data is available.
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset.
    logger.info("Loading dataset: %s", args.dataset_name)
    dataset = load_dataset(args.dataset_name)

    # 2. Count relation types on the training split and select top-K.
    type_counts = count_relation_types(dataset, split="train")
    top_types = [t for t, _ in type_counts.most_common(args.top_k)]
    allowed_types: Set[str] = set(top_types)
    logger.info(
        "Selected top-%d relation types out of %d total.",
        args.top_k, len(type_counts),
    )

    # 3. Write the relation schema file.
    schema_path = output_dir / "relation.schema"
    with open(schema_path, "w", encoding="utf-8") as f:
        for t in top_types:
            f.write(t + "\n")
    logger.info("Schema written to %s (%d types).", schema_path, len(top_types))

    # 4. Process each split.
    split_map = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}
    for split_name, filename in split_map.items():
        if split_name not in dataset:
            logger.warning("Split %r not found in dataset; skipping.", split_name)
            continue
        process_split(dataset, split_name, allowed_types, output_dir / filename)

    logger.info("Pre-processing complete.  Output directory: %s", output_dir)


if __name__ == "__main__":
    main()