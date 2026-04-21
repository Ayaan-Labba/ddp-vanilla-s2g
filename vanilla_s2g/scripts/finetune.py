"""
Fine-tuning script for the Vanilla S2G model (PLACEHOLDER).

This script will be completed after pre-training is validated.  It
mirrors the pretrain.py structure with the following differences:

- Static SSI: all schema types are included in every instance.
- Typed SEL: optional per-type entity tokens (<PER>, <LOC>, etc.).
- Early stopping on averaged strict + boundary F1.
- Loads a pre-trained checkpoint as the starting point.

Usage (planned)::

    python -m vanilla_s2g.scripts.finetune \\
        --config configs/finetune.yaml \\
        --pretrained_checkpoint outputs/pretrain/best_model \\
        --data_dir data/conll04 \\
        --schema_file data/conll04/relation.schema

TODO:
    - Static SSI collator mode (all types, all negatives).
    - Typed SEL generation with <{ent_type}> tokens.
    - Embedding initialisation from <ent> for typed tokens.
    - Dataset-specific preprocessing (CoNLL04, NYT-multi, SciERC).
    - Cross-sentence relation handling for SciERC.
    - Early stopping on avg_f1 (strict + boundary) / 2.
    - Benchmark-specific evaluation (NER + RE metrics).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    raise NotImplementedError(
        "Fine-tuning script is a placeholder.  "
        "It will be implemented after pre-training is validated."
    )


if __name__ == "__main__":
    main()