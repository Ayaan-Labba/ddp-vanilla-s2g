"""
Model module for the Vanilla S2G pipeline.

Provides the seq2seq model wrapper and FSM-based constraint decoder.

Public API
----------
- ``S2GModel``                     — Model wrapper with special-token management.
- ``ConstraintDecodingProcessor``  — FSM logits processor for constrained generation.
- ``build_constraint_processor``   — Builder function for the constraint processor.
- ``Trie``                         — Prefix tree over tokenised label names.
"""

from .constraint_decoder import (
    ConstraintDecodingProcessor,
    Trie,
    build_constraint_processor,
)
from .model import S2GModel

__all__ = [
    "S2GModel",
    "ConstraintDecodingProcessor",
    "Trie",
    "build_constraint_processor",
]