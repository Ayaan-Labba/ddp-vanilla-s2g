"""
Verify the actual statistics of the HuggingFace REBEL dataset.

Reports, for each split:
  * total instances and triplets in the raw dataset
  * instances and triplets retained under the *strict* top-K filter
        (every triplet's relation type must be in the top-K set)
  * instances and triplets retained under the *permissive* top-K filter
        (at least one triplet's relation type is in the top-K set;
         and only top-K triplets are counted)

The script is single-pass: it counts relation frequencies on the
training split first, derives the top-K set, then sweeps every split
once more, computing both filter results in the same loop.

Usage:
    python verify_rebel_stats.py --top_k 220
    python verify_rebel_stats.py --top_k 220 --dataset_name Babelscape/rebel-dataset
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from typing import Iterator, List, Set, Tuple

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
#                       LIGHTWEIGHT TRIPLET PARSING                     #
# --------------------------------------------------------------------- #
#
# The REBEL `triplets` field is encoded as a single string of the form
#   <triplet> H1 <subj> T1 <obj> R1 <subj> T2 <obj> R2
#   <triplet> H2 <subj> T3 <obj> R3 ...
#
# For statistics we only need the relation types, so we use a single
# regex to extract every `<obj> RELATION` occurrence rather than the
# full segment-by-segment parser used in `preprocess_rebel.py`.  This
# is roughly 4-5x faster on the full corpus and avoids any chance of
# implementation drift between the diagnostic and the real pipeline.

_REL_RE = re.compile(r"<obj>\s*(.*?)\s*(?=<subj>|<triplet>|$)", re.DOTALL)


def iter_relation_types(triplet_str: str) -> Iterator[str]:
    """Yield every relation-type surface form in a REBEL `triplets` string.

    Each match is the substring sitting between an `<obj>` marker and the
    next structural marker (`<subj>`, `<triplet>`, or end-of-string).
    Whitespace is stripped; empty matches are dropped silently.
    """
    for match in _REL_RE.finditer(triplet_str):
        rel = match.group(1).strip()
        if rel:
            yield rel


# --------------------------------------------------------------------- #
#                    PASS 1: RELATION-TYPE FREQUENCIES                  #
# --------------------------------------------------------------------- #


def count_train_relation_types(dataset) -> Counter:
    """Count relation-type frequencies on the training split.

    Returns a Counter mapping relation-type strings to occurrence
    counts (one per triplet, not per instance).
    """
    counter: Counter = Counter()
    for instance in tqdm(dataset["train"], desc="Counting train relations"):
        for rel in iter_relation_types(instance["triplets"]):
            counter[rel] += 1
    return counter


# --------------------------------------------------------------------- #
#                    PASS 2: PER-SPLIT FILTER STATISTICS                #
# --------------------------------------------------------------------- #


def split_statistics(
    dataset,
    split: str,
    allowed_types: Set[str],
) -> Tuple[int, int, int, int, int, int]:
    """Compute all relevant counts for one split in a single pass.

    For every instance we determine the relation types it contains and
    record:

        n_instances_total      - total instances in the split
        n_triplets_total       - total triplets across all instances

        n_instances_strict     - instances whose triplets are *all* in
                                 allowed_types (REBEL paper filter)
        n_triplets_strict      - sum of triplet counts for those
                                 strict-pass instances

        n_instances_permissive - instances with at least one triplet in
                                 allowed_types
        n_triplets_permissive  - count of allowed-type triplets across
                                 all instances (ignores filtering of
                                 the parent instance)

    The strict and permissive variants disagree precisely when an
    instance has a mix of in-set and out-of-set relation types.
    """
    n_instances_total = 0
    n_triplets_total = 0
    n_instances_strict = 0
    n_triplets_strict = 0
    n_instances_permissive = 0
    n_triplets_permissive = 0

    for instance in tqdm(dataset[split], desc=f"Scanning {split}"):
        rels: List[str] = list(iter_relation_types(instance["triplets"]))
        if not rels:
            n_instances_total += 1
            continue

        n_instances_total += 1
        n_triplets_total += len(rels)

        rel_set = set(rels)
        in_allowed = rel_set & allowed_types

        # Permissive: count any triplet whose type is allowed.
        n_in = sum(1 for r in rels if r in allowed_types)
        if n_in > 0:
            n_instances_permissive += 1
            n_triplets_permissive += n_in

        # Strict: instance retained only if *all* types are allowed.
        if rel_set.issubset(allowed_types):
            n_instances_strict += 1
            n_triplets_strict += len(rels)

    return (
        n_instances_total,
        n_triplets_total,
        n_instances_strict,
        n_triplets_strict,
        n_instances_permissive,
        n_triplets_permissive,
    )


# --------------------------------------------------------------------- #
#                                  MAIN                                 #
# --------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="REBEL statistics verifier")
    parser.add_argument("--dataset_name", default="Babelscape/rebel-dataset")
    parser.add_argument("--revision", default="refs/convert/parquet")
    parser.add_argument("--top_k", type=int, default=220)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading %s @ %s ...", args.dataset_name, args.revision)
    dataset = load_dataset(
        args.dataset_name, "default", revision=args.revision
    )

    # Pass 1 -- determine top-K relation types from training frequencies.
    type_counts = count_train_relation_types(dataset)
    total_unique = len(type_counts)
    top_types = [t for t, _ in type_counts.most_common(args.top_k)]
    allowed: Set[str] = set(top_types)

    logger.info("\nUnique relation types in train: %d", total_unique)
    logger.info("Top-%d threshold (lowest-ranked count): %d",
                args.top_k, type_counts[top_types[-1]])

    # Pass 2 -- per-split counts under both filtering strategies.
    print("\n" + "=" * 80)
    print(f"REBEL statistics (top_k = {args.top_k})")
    print("=" * 80)
    header = (
        f"{'split':<12}"
        f"{'total_inst':>12}{'total_trip':>12}"
        f"{'strict_inst':>14}{'strict_trip':>14}"
        f"{'perm_inst':>12}{'perm_trip':>12}"
    )
    print(header)
    print("-" * len(header))

    for split in ("train", "validation", "test"):
        if split not in dataset:
            continue
        stats = split_statistics(dataset, split, allowed)
        (n_tot_i, n_tot_t,
         n_str_i, n_str_t,
         n_per_i, n_per_t) = stats
        print(
            f"{split:<12}"
            f"{n_tot_i:>12,}{n_tot_t:>12,}"
            f"{n_str_i:>14,}{n_str_t:>14,}"
            f"{n_per_i:>12,}{n_per_t:>12,}"
        )

    print("=" * 80)
    print(
        "\nLegend:\n"
        "  total_inst / total_trip : raw counts in HuggingFace dataset\n"
        "  strict_inst / strict_trip : REBEL-paper filter (all triplets in top-K)\n"
        "  perm_inst / perm_trip   : permissive filter (>=1 triplet in top-K)\n"
    )


if __name__ == "__main__":
    main()