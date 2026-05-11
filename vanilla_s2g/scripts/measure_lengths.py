"""
Length-budget scan for the Vanilla S2G model.

Reads every instance in train, val, and test, constructs the worst-case
encoder input (all gold positives + uniformly sampled negatives filling
the SSI to ``max_types_in_prompt``) and the gold SEL target, tokenises
both *without truncation*, and reports a percentile table (p50, p75,
p90, p95, p99, max) of tokenised lengths per split and overall.

The p99 values are used for the YAML suggestions: they exclude extreme
outliers while still covering the vast majority of the corpus.  The max
column is included for reference.  Setting
``cfg.tokenization.max_source_length`` and
``cfg.tokenization.max_target_length`` to at least the overall p99
(rounded up) guarantees truncation for at most ~1 % of instances.

Only a tokeniser is loaded (no model), so the scan is cheap to run
even on a CPU-only machine.

Usage::

    python -m vanilla_s2g.scripts.measure_lengths --config configs/pretrain.yaml

CLI dotlist overrides work as elsewhere.  For example, to probe a
tighter prompt budget without editing the YAML::

    python -m vanilla_s2g.scripts.measure_lengths --config configs/pretrain.yaml \\
        ssi.max_types_in_prompt=15

The scan reads only ``cfg.model.name``, ``cfg.data.data_dir``,
``cfg.data.schema_file``, ``cfg.ssi.max_types_in_prompt``, and
``cfg.train.seed``.  All other config fields are ignored.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

from vanilla_s2g.data import S2GDataset
from vanilla_s2g.linearisation import (
    add_special_tokens_to_tokenizer,
    build_encoder_input,
)
from vanilla_s2g.scripts.config_utils import load_config, load_schema

logger = logging.getLogger(__name__)


# Percentile points reported in every scan table.  100 is the maximum.
_PCTS: tuple = (50, 75, 90, 95, 99, 100)
_PCT_LABELS: tuple = tuple("max" if p == 100 else f"p{p}" for p in _PCTS)


@dataclass
class SplitStats:
    """Percentile breakdown of tokenised source and target lengths for one split.

    Attributes:
        src: Mapping from percentile (one of :data:`_PCTS`) to source token count.
        tgt: Mapping from percentile (one of :data:`_PCTS`) to target token count.
    """

    src: Dict[int, int]  # e.g. {50: 47, 75: 68, 90: 89, 95: 103, 99: 138, 100: 883}
    tgt: Dict[int, int]


# ===================================================================== #
#                       BUDGET-MODE SSI SAMPLING                         #
# ===================================================================== #


def _sample_budget_ssi(
    schema: List[str],
    instance_types: List[str],
    max_types_in_prompt: int,
    rng: random.Random,
) -> List[str]:
    """Construct the worst-case SSI for one instance under budget mode.

    Mirrors the budget-mode branch of ``S2GCollator._sample_types`` and
    the helper in ``evaluate.py``: every gold positive is included, and
    negatives are drawn uniformly from the negative pool to fill the
    remaining budget.  Positives are never truncated.
    """
    instance_set = set(instance_types)
    negative_pool = [t for t in schema if t not in instance_set]
    n_neg = max(0, max_types_in_prompt - len(instance_types))
    n_neg = min(n_neg, len(negative_pool))
    sampled = rng.sample(negative_pool, n_neg) if n_neg > 0 else []
    return list(instance_types) + sampled


# ===================================================================== #
#                              SCAN                                      #
# ===================================================================== #


def _scan_split(
    split_name: str,
    dataset: S2GDataset,
    tokenizer,
    schema: List[str],
    max_types_in_prompt: int,
    seed: int,
    batch_size: int = 256,
) -> SplitStats:
    """Return percentile breakdowns of source and target token counts for one split.

    All instance lengths are collected into in-memory lists during the scan,
    then percentiles are computed with NumPy after the full pass completes.
    Peak extra RAM is two int32 arrays of ``len(dataset)`` elements each —
    approximately 6 MB for 784 K REBEL instances.

    Tokenisation is performed without truncation or padding so that the
    measured lengths reflect the true token count for each instance.
    Tokeniser calls are still batched (default 256 instances per call) to
    amortise the per-call overhead.

    Args:
        split_name:           Human-readable name used in the progress bar.
        dataset:              Loaded :class:`S2GDataset` for the split.
        tokenizer:            HuggingFace tokeniser with S2G special tokens.
        schema:               Full list of relation-type strings.
        max_types_in_prompt:  SSI budget cap (mirrors budget-mode collation).
        seed:                 RNG seed for deterministic negative sampling.
        batch_size:           Number of instances tokenised per call.

    Returns:
        :class:`SplitStats` with source and target percentile dicts.
    """
    rng = random.Random(seed)
    src_lens: List[int] = []
    tgt_lens: List[int] = []
    n = len(dataset)

    for start in tqdm(range(0, n, batch_size), desc=f"Scanning {split_name}"):
        end = min(start + batch_size, n)

        encoder_inputs: List[str] = []
        sel_targets: List[str] = []
        for i in range(start, end):
            inst = dataset[i]
            ssi_types = _sample_budget_ssi(
                schema, inst["types"], max_types_in_prompt, rng,
            )
            encoder_inputs.append(
                build_encoder_input(
                    ssi_types, inst["text"], random_prompt=False,
                )
            )
            sel_targets.append(inst["sel"])

        # No truncation, no padding — we want true lengths.
        src_ids = tokenizer(encoder_inputs, add_special_tokens=True)["input_ids"]
        tgt_ids = tokenizer(sel_targets, add_special_tokens=True)["input_ids"]

        src_lens.extend(len(ids) for ids in src_ids)
        tgt_lens.extend(len(ids) for ids in tgt_ids)

    # Compute all percentile points in a single vectorised pass.
    src_arr = np.array(src_lens, dtype=np.int32)
    tgt_arr = np.array(tgt_lens, dtype=np.int32)

    def _pct_dict(arr: np.ndarray) -> Dict[int, int]:
        # method='lower' ensures each returned value is an actual observed
        # length, not a float interpolation between two adjacent values.
        return {
            p: int(np.percentile(arr, p, method="lower"))
            for p in _PCTS
        }

    return SplitStats(src=_pct_dict(src_arr), tgt=_pct_dict(tgt_arr))


# ===================================================================== #
#                              MAIN                                      #
# ===================================================================== #


def _round_up(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*."""
    return ((value + multiple - 1) // multiple) * multiple


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config()
    logger.info("Configuration loaded: %s", cfg.config_path)

    if cfg.ssi.max_types_in_prompt is None:
        raise ValueError(
            "ssi.max_types_in_prompt must be set to scan budget-mode SSI "
            "lengths.  Set it in the YAML or on the CLI with "
            "'ssi.max_types_in_prompt=<int>'."
        )
    cap = int(cfg.ssi.max_types_in_prompt)
    seed = cfg.train.seed

    # ---- Tokeniser only (no model load) ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    num_added = add_special_tokens_to_tokenizer(tokenizer)
    logger.info(
        "Tokenizer loaded: %s (+%d S2G special tokens)",
        cfg.model.name, num_added,
    )

    # ---- Schema ----
    schema = load_schema(cfg.data.schema_file)
    logger.info(
        "Schema loaded: %d relation types; SSI cap = %d.",
        len(schema), cap,
    )

    # ---- Scan every available split ----
    data_dir = Path(cfg.data.data_dir)
    splits = {
        "train": data_dir / "train.jsonl",
        "val":   data_dir / "val.jsonl",
        "test":  data_dir / "test.jsonl",
    }

    per_split: Dict[str, SplitStats] = {}
    for name, path in splits.items():
        if not path.exists():
            logger.warning("Skipping %s: %s does not exist.", name, path)
            continue
        dataset = S2GDataset(path, seed=seed)
        per_split[name] = _scan_split(
            name, dataset, tokenizer, schema, cap, seed,
        )

    if not per_split:
        raise RuntimeError("No splits were scanned; check cfg.data.data_dir.")

    # ---- Report ----
    # Compute an overall row as the element-wise max across splits.  This is
    # conservative: e.g. the overall p99 is the largest p99 across all splits,
    # not the p99 of the pooled distribution.  For setting truncation budgets,
    # conservative is the right direction.
    split_names = list(per_split.keys())

    def _overall(field: str) -> Dict[int, int]:
        return {
            p: max(getattr(per_split[s], field)[p] for s in split_names)
            for p in _PCTS
        }

    overall_src = _overall("src")
    overall_tgt = _overall("tgt")

    # Column width: each percentile value uses 8 characters.
    col_w = 8
    header = "".join(f"{lbl:>{col_w}}" for lbl in _PCT_LABELS)

    def _row(stats_dict: Dict[int, int]) -> str:
        return "".join(f"{stats_dict[p]:>{col_w}d}" for p in _PCTS)

    sep = "=" * (10 + col_w * len(_PCTS))
    thin = "-" * (10 + col_w * len(_PCTS))

    logger.info(sep)
    logger.info(
        "Source token-length percentiles  (max_types_in_prompt = %d)", cap
    )
    logger.info(thin)
    logger.info(f"{'split':<10}{header}")
    for name, stats in per_split.items():
        logger.info(f"{name:<10}{_row(stats.src)}")
    logger.info(thin)
    logger.info(f"{'overall':<10}{_row(overall_src)}")
    logger.info(sep)

    logger.info(sep)
    logger.info(
        "Target token-length percentiles  (max_types_in_prompt = %d)", cap
    )
    logger.info(thin)
    logger.info(f"{'split':<10}{header}")
    for name, stats in per_split.items():
        logger.info(f"{name:<10}{_row(stats.tgt)}")
    logger.info(thin)
    logger.info(f"{'overall':<10}{_row(overall_tgt)}")
    logger.info(sep)

    logger.info(
        "Suggested YAML values based on overall p99 "
        "(rounded up to nearest 50 for safety):\n"
        "  tokenization.max_source_length: %d\n"
        "  tokenization.max_target_length: %d",
        _round_up(overall_src[99], 50),
        _round_up(overall_tgt[99], 50),
    )


if __name__ == "__main__":
    main()