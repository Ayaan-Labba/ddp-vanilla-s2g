"""
Data module for the Vanilla S2G pipeline.

Provides dataset loading, dynamic data collation with type sampling,
and a preprocessing script for the REBEL dataset.

Public API
----------
- ``S2GDataset``   — PyTorch Dataset backed by a JSONL file.
- ``S2GCollator``  — Data collator with dynamic SSI construction.
"""

from .collator import S2GCollator
from .dataset import S2GDataset

__all__ = [
    "S2GDataset",
    "S2GCollator",
]