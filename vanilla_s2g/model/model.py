"""
S2G Model — seq2seq wrapper around a HuggingFace encoder-decoder model.

This module manages the lifecycle of the underlying language model:

1. **Initialisation** — Loads a Flan-T5 checkpoint, registers the S2G
   special tokens, and resizes the embedding matrix.

2. **Typed entity tokens** (fine-tuning only) — Registers per-type
   entity tokens (``<PER>``, ``<LOC>``, …) and initialises each new
   embedding as a copy of the pre-trained ``<ent>`` embedding.

3. **Generation** — Provides a ``generate`` method that optionally
   activates constraint decoding via an FSM-based ``LogitsProcessor``.

The wrapper deliberately does **not** subclass ``nn.Module``.  It is a
plain Python class that holds a reference to the HuggingFace model and
tokeniser, because PyTorch Lightning's ``LightningModule`` (used in the
training script) already manages the ``nn.Module`` lifecycle.  Keeping
the wrapper non-``Module`` avoids double-wrapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from vanilla_s2g.linearisation.special_tokens import (
    SPECIAL_TOKENS,
    SpecialTokens,
    add_special_tokens_to_tokenizer,
    get_token_ids,
)

logger = logging.getLogger(__name__)


class S2GModel:
    """Wrapper around a HuggingFace seq2seq model with S2G special tokens.

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint path.
        special_tokens:     Token registry (defaults to the module singleton).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-large",
        special_tokens: Optional[SpecialTokens] = None,
    ) -> None:
        self.special_tokens = special_tokens or SPECIAL_TOKENS

        logger.info("Loading tokenizer from %s", model_name_or_path)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )

        logger.info("Loading model from %s", model_name_or_path)
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
        )

        # Register special tokens and resize embeddings.
        num_added = add_special_tokens_to_tokenizer(
            self.tokenizer, self.model, self.special_tokens
        )
        logger.info("Added %d special tokens to tokenizer.", num_added)

        # Cache token IDs for fast lookup.
        self.token_ids: Dict[str, int] = get_token_ids(
            self.tokenizer, self.special_tokens
        )

    # ------------------------------------------------------------------ #
    #  Typed entity tokens (fine-tuning)                                  #
    # ------------------------------------------------------------------ #

    def add_entity_type_tokens(self, entity_types: List[str]) -> int:
        """Register typed entity tokens and initialise from ``<ent>``.

        For each entity type string (e.g. ``"PER"``), a new special token
        ``<PER>`` is added.  Its embedding is initialised as a **copy** of
        the pre-trained ``<ent>`` embedding so that the model starts from
        a meaningful representation.

        Args:
            entity_types: List of entity-type strings to register.

        Returns:
            Number of new tokens actually added.
        """
        if not entity_types:
            return 0

        # Retrieve the current <ent> embedding *before* resizing.
        ent_id = self.token_ids["ent_start"]
        ent_embedding = (
            self.model.get_input_embeddings().weight[ent_id].clone().detach()
        )

        # Build typed tokens like <PER>, <LOC>, etc.
        typed_tokens = [f"<{et}>" for et in entity_types]
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": typed_tokens}
        )

        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

            # Copy <ent> embedding into each new token slot.
            with torch.no_grad():
                embed = self.model.get_input_embeddings()
                for token_str in typed_tokens:
                    token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                    embed.weight[token_id] = ent_embedding.clone()

            logger.info(
                "Added %d typed entity tokens, initialised from <ent>.", num_added
            )

        return num_added

    # ------------------------------------------------------------------ #
    #  Generation                                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        constraint_decoding: bool = False,
        source_ids: Optional[torch.Tensor] = None,
        relation_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run autoregressive generation with optional constraint decoding.

        When *constraint_decoding* is ``True``, an FSM-based
        ``LogitsProcessor`` is injected that restricts the decoder
        vocabulary at each step.  This requires *source_ids* (for the
        source-copy constraint) and *relation_types* (for the label trie).

        Additional keyword arguments are forwarded to the HuggingFace
        ``model.generate()`` call (e.g. ``num_beams``, ``max_length``).

        Args:
            input_ids:           Encoder input token IDs ``(batch, src_len)``.
            attention_mask:      Encoder attention mask ``(batch, src_len)``.
            constraint_decoding: Whether to activate FSM constraints.
            source_ids:          Encoder input IDs for source-copy constraint
                                 ``(batch, src_len)``.  Required when
                                 *constraint_decoding* is ``True``.
            relation_types:      List of all valid relation-type strings.
                                 Required when *constraint_decoding* is ``True``.
            **kwargs:            Forwarded to ``model.generate()``.

        Returns:
            Generated token IDs ``(batch, tgt_len)``.
        """
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }

        if constraint_decoding:
            if source_ids is None or relation_types is None:
                raise ValueError(
                    "constraint_decoding requires both source_ids and "
                    "relation_types to be provided."
                )

            from vanilla_s2g.model.constraint_decoder import (
                build_constraint_processor,
            )

            processor = build_constraint_processor(
                tokenizer=self.tokenizer,
                source_ids=source_ids,
                relation_types=relation_types,
                special_tokens=self.special_tokens,
            )
            gen_kwargs["logits_processor"] = [processor]

        return self.model.generate(**gen_kwargs)

    # ------------------------------------------------------------------ #
    #  Serialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save both model and tokeniser to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Model and tokenizer saved to %s", path)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        special_tokens: Optional[SpecialTokens] = None,
    ) -> "S2GModel":
        """Load a previously saved S2G model.

        This avoids re-adding special tokens (they are already in the
        saved tokeniser vocabulary) but still refreshes the token ID cache.
        """
        instance = cls.__new__(cls)
        instance.special_tokens = special_tokens or SPECIAL_TOKENS

        path_str = str(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path_str)
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(path_str)
        instance.token_ids = get_token_ids(
            instance.tokenizer, instance.special_tokens
        )

        logger.info("Loaded S2G model from %s", path)
        return instance