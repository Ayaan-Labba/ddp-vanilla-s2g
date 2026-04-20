"""
Special token registry for the Vanilla S2G model.

Defines all structural markers used in the SSI (encoder input) and SEL
(decoder output). Provides utilities for adding tokens to a HuggingFace
tokeniser and retrieving their integer IDs after registration.

Design note: The registry is a frozen dataclass singleton so that token
strings are defined in exactly one place. Every other module imports
SPECIAL_TOKENS rather than hard-coding strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class SpecialTokens:
    """Immutable registry of all special tokens used by the S2G model.

    Attributes:
        type_prompt:  Introduces one relation type in the SSI.
        text_start:   Delimiter between the SSI prefix and the raw text.
        ent_start:    Introduces the surface-text span of an entity in SEL.
        rel_open:     Opens a labelled relation in SEL.
        tail_start:   Introduces the surface-text span of a tail entity in SEL.
        reject_type:  Rejection token for relation types present in schema
                      but absent from the text.
    """

    # Encoder-side tokens (SSI)
    type_prompt: str = "<type>"
    text_start: str = "<text>"

    # Decoder-side tokens (SEL)
    ent_start: str = "<ent>"
    rel_open: str = "<rel>"
    tail_start: str = "<tail>"
    reject_type: str = "<null>"

    # ----- Derived collections -----

    @property
    def encoder_tokens(self) -> List[str]:
        """Tokens that appear only on the encoder (input) side."""
        return [self.type_prompt, self.text_start]

    @property
    def decoder_tokens(self) -> List[str]:
        """Tokens that appear only on the decoder (output) side."""
        return [self.ent_start, self.rel_open, self.tail_start, self.reject_type]

    @property
    def all_tokens(self) -> List[str]:
        """All special tokens in a stable, deterministic order."""
        return self.encoder_tokens + self.decoder_tokens

    def as_dict(self) -> Dict[str, str]:
        """Map from constant name to token string."""
        return {
            "type_prompt": self.type_prompt,
            "text_start": self.text_start,
            "ent_start": self.ent_start,
            "rel_open": self.rel_open,
            "tail_start": self.tail_start,
            "reject_type": self.reject_type,
        }


# Module-level singleton used by all other modules.
SPECIAL_TOKENS = SpecialTokens()


def add_special_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    model: Optional[PreTrainedModel] = None,
    special_tokens: Optional[SpecialTokens] = None,
) -> int:
    """Register S2G special tokens with *tokenizer* and optionally resize *model* embeddings.

    Args:
        tokenizer: A HuggingFace tokeniser instance.
        model:     If provided, its embedding matrix is resized to accommodate
                   the new tokens.
        special_tokens: Token registry to use.  Defaults to ``SPECIAL_TOKENS``.

    Returns:
        The number of tokens actually added (0 if they were already present).
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens.all_tokens}
    )

    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return num_added


def get_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    special_tokens: Optional[SpecialTokens] = None,
) -> Dict[str, int]:
    """Return a mapping from constant name to integer token ID.

    Must be called *after* :func:`add_special_tokens_to_tokenizer`.

    Args:
        tokenizer:     HuggingFace tokeniser with special tokens registered.
        special_tokens: Token registry to use.  Defaults to ``SPECIAL_TOKENS``.

    Returns:
        ``{"type_prompt": 32100, "text_start": 32101, ...}``
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    return {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in special_tokens.as_dict().items()
    }