"""
S2G Dataset — loads preprocessed JSONL instances for training and evaluation.

This is a thin wrapper around a JSONL file.  Each line is a JSON object
in the S2G standardised format (see ``preprocess_rebel.py`` for details).
The dataset performs no tokenisation or SSI construction — all dynamic
processing is deferred to :class:`~vanilla_s2g.data.collator.S2GCollator`
so that stochastic sampling (positive/negative types) is refreshed each
epoch rather than baked into the dataset.

An optional *subset_fraction* parameter allows validation on a random
subset of the data (used for ``val_percent_check`` during training).

Memory strategy
~~~~~~~~~~~~~~~
Raw JSON strings are stored instead of parsed Python dicts.  Parsing is
deferred to ``__getitem__``, which is called inside DataLoader worker
processes.  This reduces RAM from ~7–8 GB (parsed dicts for 784 K REBEL
instances) to ~300–500 MB (raw strings), making the dataset compatible
with memory-constrained environments such as Google Colab (~12.7 GB RAM).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class S2GDataset(Dataset):
    """Lazy-parsed dataset backed by a JSONL file.

    Raw JSON lines are held in memory as strings; each call to
    ``__getitem__`` parses the requested line on the fly.  This keeps
    RAM usage proportional to the *text size* of the JSONL file rather
    than the *Python-object size* of the parsed dicts, which is
    typically 15–20× larger.

    Args:
        filepath:        Path to a ``.jsonl`` file in S2G format.
        subset_fraction: If set to a value in ``(0, 1]``, only this
                         fraction of the instances is retained (sampled
                         deterministically for reproducibility).
        seed:            Random seed used when *subset_fraction* is active.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        subset_fraction: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        logger.info("Loading dataset from %s", filepath)

        # ── Store raw JSON strings instead of parsed dicts ────────────
        # Each string is ~300–600 bytes of contiguous character data.
        # The equivalent parsed dict would be ~8,000–12,000 bytes of
        # scattered Python objects (dicts, lists, strings, ints).
        self._raw_lines: List[str] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._raw_lines.append(line)

        logger.info(
            "Loaded %d instances from %s (raw strings, ~%.0f MB)",
            len(self._raw_lines),
            filepath.name,
            sum(len(l) for l in self._raw_lines) / 1e6,
        )

        # Apply optional subsetting.
        if subset_fraction is not None and 0 < subset_fraction < 1:
            rng = random.Random(seed)
            n = max(1, int(len(self._raw_lines) * subset_fraction))
            self._raw_lines = rng.sample(self._raw_lines, n)
            logger.info(
                "Subsetted to %d instances (%.0f%%)",
                n, subset_fraction * 100,
            )

    def __len__(self) -> int:
        return len(self._raw_lines)

    def __getitem__(self, idx: int) -> Dict:
        """Parse and return the instance at *idx*.

        JSON parsing is deferred to this call so that it happens inside
        DataLoader worker processes rather than in the main process at
        init time.  ``json.loads`` on a ~500-byte string takes ~5–10 µs,
        which is negligible compared to the collator's tokenisation cost.

        The returned dict contains keys: ``text``, ``tokens``,
        ``entities``, ``relations``, ``types``, ``sel``.
        """
        return json.loads(self._raw_lines[idx])