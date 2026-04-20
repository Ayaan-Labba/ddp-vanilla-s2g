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
    """Memory-mapped dataset backed by a JSONL file.

    All instances are loaded into memory at construction time.  For the
    REBEL pre-training set (~784 K instances), this consumes roughly
    2–4 GB of RAM, which is acceptable for modern training machines.

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
        self.instances: List[Dict] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instances.append(json.loads(line))

        logger.info("Loaded %d instances from %s", len(self.instances), filepath.name)

        # Apply optional subsetting.
        if subset_fraction is not None and 0 < subset_fraction < 1:
            rng = random.Random(seed)
            n = max(1, int(len(self.instances) * subset_fraction))
            self.instances = rng.sample(self.instances, n)
            logger.info(
                "Subsetted to %d instances (%.0f%%)",
                n, subset_fraction * 100,
            )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict:
        """Return the raw instance dict at *idx*.

        The dict contains keys: ``text``, ``tokens``, ``entities``,
        ``relations``, ``types``, ``sel``.
        """
        return self.instances[idx]