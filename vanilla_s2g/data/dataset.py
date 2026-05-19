"""
S2G Dataset — memory-mapped JSONL reader for training and evaluation.

Instances are accessed via a NumPy offset index into a read-only mmap of the
JSONL file.  This avoids storing 3M+ Python string objects in a List[str],
which would consume ~5 GB of heap and multiply by the number of DataLoader
worker processes via fork copy-on-write.

Memory profile for 3M instances (vs List[str]):
    Offset index  :  3M × 16 bytes  ≈  48 MB   (NumPy, fork-safe)
    File pages    :  OS-managed, shared read-only across all workers
    Per-worker CoW:  ~a few KB      (only mmap/numpy Python object headers)

    Total (main + 4 workers):  ~50 MB  vs  ~32 GB with List[str]

Spawn-mode compatibility
~~~~~~~~~~~~~~~~~~~~~~~~
mmap objects are not picklable.  __getstate__ / __setstate__ strip and
re-open the mmap so the dataset survives DataLoader worker pickling under
the 'spawn' start method as well as the default 'fork'.
"""

from __future__ import annotations

import json
import logging
import mmap
from pathlib import Path
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Chunk size for the newline-scan pass during __init__.
# 64 MB keeps peak init memory well below 100 MB regardless of file size.
_SCAN_CHUNK_BYTES = 64 * 1024 * 1024


class S2GDataset(Dataset):
    """Memory-mapped dataset backed by a JSONL file.

    Line byte-offsets are stored in a compact NumPy array.  Each call to
    ``__getitem__`` slices the mmap directly and calls ``json.loads`` on the
    raw bytes — no Python string objects are retained between calls.

    Args:
        filepath:        Path to a ``.jsonl`` file in S2G format.
        subset_fraction: If in ``(0, 1)``, retain only this fraction of
                         instances, sampled deterministically.
        seed:            RNG seed used when *subset_fraction* is active.
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

        self._filepath = filepath

        logger.info("Indexing %s …", filepath)
        self._offsets = _build_offset_index(filepath)
        logger.info(
            "Indexed %d instances from %s  (offset table: %.1f MB)",
            len(self._offsets),
            filepath.name,
            self._offsets.nbytes / 1e6,
        )

        if subset_fraction is not None and 0.0 < subset_fraction < 1.0:
            rng = np.random.default_rng(seed)
            n = max(1, int(len(self._offsets) * subset_fraction))
            indices = np.sort(
                rng.choice(len(self._offsets), size=n, replace=False)
            )
            self._offsets = self._offsets[indices]
            logger.info(
                "Subsetted to %d instances (%.0f%%)", n, subset_fraction * 100
            )

        self._file, self._mmap = _open_mmap(self._filepath)

    # ------------------------------------------------------------------ #
    # Dataset interface                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        """Return the parsed instance at *idx*.

        Slices the mmap directly — no copy of the full file is held in
        Python heap memory.  ``json.loads`` accepts ``bytes`` since Py 3.6.
        """
        start = int(self._offsets[idx, 0])
        end   = int(self._offsets[idx, 1])
        return json.loads(self._mmap[start:end])

    # ------------------------------------------------------------------ #
    # Pickle support (DataLoader spawn mode)                               #
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict:
        """Strip the un-picklable mmap before serialisation."""
        state = self.__dict__.copy()
        del state["_mmap"]
        del state["_file"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Re-open the mmap after deserialisation in a worker process."""
        self.__dict__.update(state)
        self._file, self._mmap = _open_mmap(self._filepath)

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def __del__(self) -> None:
        try:
            self._mmap.close()
            self._file.close()
        except Exception:
            pass


# ======================================================================= #
#  Module-level helpers                                                    #
# ======================================================================= #


def _open_mmap(filepath: Path) -> tuple[object, mmap.mmap]:
    """Open *filepath* and return ``(file_handle, read-only mmap)``."""
    f = open(filepath, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return f, m


def _build_offset_index(filepath: Path) -> np.ndarray:
    """Scan *filepath* and return an ``(N, 2)`` int64 array of byte offsets.

    Each row is ``[line_start, line_end)`` in bytes (end excludes the newline).
    The file is processed in fixed-size chunks; peak memory during this call
    is bounded by ``_SCAN_CHUNK_BYTES`` regardless of file size.

    Returns:
        NumPy array of shape ``(N, 2)`` and dtype ``int64``.
    """
    starts: list[int] = []
    ends:   list[int] = []

    chunk_base  = 0   # byte position of the start of the current chunk
    line_start  = 0   # byte position of the start of the current line

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(_SCAN_CHUNK_BYTES)
            if not chunk:
                # File exhausted: flush any unterminated final line.
                if line_start < chunk_base:
                    starts.append(line_start)
                    ends.append(chunk_base)
                break

            # Vectorised newline scan — far faster than a Python loop.
            arr = np.frombuffer(chunk, dtype=np.uint8)
            for nl_rel in np.where(arr == 10)[0]:          # 10 == ord('\n')
                nl_abs = chunk_base + int(nl_rel)
                if nl_abs > line_start:                     # skip blank lines
                    starts.append(line_start)
                    ends.append(nl_abs)
                line_start = nl_abs + 1

            chunk_base += len(chunk)

    if not starts:
        return np.empty((0, 2), dtype=np.int64)

    return np.column_stack([
        np.array(starts, dtype=np.int64),
        np.array(ends,   dtype=np.int64),
    ])