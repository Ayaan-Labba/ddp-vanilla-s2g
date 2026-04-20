"""
Linearisation module for the Vanilla S2G model.

This package encapsulates the complete representation of structured
knowledge-graph information as flat token sequences.  It is the single
point of change when experimenting with alternative linearisation formats.

Public API
----------
- ``SPECIAL_TOKENS``               — Token registry singleton.
- ``add_special_tokens_to_tokenizer`` — Register tokens with a HF tokeniser.
- ``get_token_ids``                — Retrieve integer IDs after registration.
- ``build_ssi_prefix``             — Build the SSI prefix string.
- ``build_encoder_input``          — Build SSI + ``<text>`` + raw text.
- ``organize_by_entity``           — Convert flat data to entity-centric blocks.
- ``filter_entity_blocks``         — Prune relations by allowed types.
- ``build_sel``                    — Build the SEL target string.
- ``parse_sel``                    — Parse a generated SEL string.
- ``extract_triplets``             — Flatten entity blocks to triplets.
"""

from .sel import (
    build_sel,
    extract_triplets,
    filter_entity_blocks,
    organize_by_entity,
    parse_sel,
)
from .special_tokens import (
    SPECIAL_TOKENS,
    SpecialTokens,
    add_special_tokens_to_tokenizer,
    get_token_ids,
)
from .ssi import build_encoder_input, build_ssi_prefix

__all__ = [
    # special_tokens
    "SPECIAL_TOKENS",
    "SpecialTokens",
    "add_special_tokens_to_tokenizer",
    "get_token_ids",
    # ssi
    "build_ssi_prefix",
    "build_encoder_input",
    # sel
    "organize_by_entity",
    "filter_entity_blocks",
    "build_sel",
    "parse_sel",
    "extract_triplets",
]