"""
Structural Schema Instructor (SSI) construction for the S2G encoder input.

The SSI is a prefix prepended to the raw text that enumerates the relation
types currently in scope for a given instance.  Each relation type is
introduced by a ``<type>`` special token.  The full encoder input is::

    <type> label_1 <type> label_2 ... <text> raw sentence tokens

This module is deliberately string-level only — tokenisation into subword
IDs happens downstream in the data collator.  Keeping the two stages
separate makes it easy to swap in a different SSI format without touching
the collation or training logic.
"""

from __future__ import annotations

import random as _random
from typing import List, Optional

from .special_tokens import SPECIAL_TOKENS, SpecialTokens


def build_ssi_prefix(
    relation_types: List[str],
    random_prompt: bool = False,
    special_tokens: Optional[SpecialTokens] = None,
) -> str:
    """Build the SSI prefix string from a list of relation types.

    Each type is preceded by the ``<type>`` token.  Types are sorted
    alphabetically by default so that the prefix is deterministic; set
    *random_prompt* to ``True`` to shuffle the order (useful as a data
    augmentation during training).

    Args:
        relation_types:  Relation-type label strings to include.
        random_prompt:   If ``True``, shuffle type order.  Otherwise sort
                         alphabetically.
        special_tokens:  Token registry (defaults to the module singleton).

    Returns:
        SSI prefix string, e.g. ``"<type> place of birth <type> president of"``.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    types = list(relation_types)

    if random_prompt:
        _random.shuffle(types)
    else:
        types.sort()

    parts = [f"{special_tokens.type_prompt} {t}" for t in types]
    return " ".join(parts)


def build_encoder_input(
    relation_types: List[str],
    text: str,
    random_prompt: bool = False,
    special_tokens: Optional[SpecialTokens] = None,
) -> str:
    """Build the complete encoder input: SSI prefix + ``<text>`` + raw text.

    Args:
        relation_types:  Relation-type label strings in scope.
        text:            Raw input sentence.
        random_prompt:   If ``True``, shuffle type order in the SSI.
        special_tokens:  Token registry (defaults to the module singleton).

    Returns:
        Full encoder input string ready for tokenisation.

    Example::

        >>> build_encoder_input(
        ...     ["place of birth", "president of"],
        ...     "Barack Obama was born in Honolulu",
        ... )
        '<type> place of birth <type> president of <text> Barack Obama was born in Honolulu'
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    ssi = build_ssi_prefix(relation_types, random_prompt, special_tokens)
    return f"{ssi} {special_tokens.text_start} {text}"