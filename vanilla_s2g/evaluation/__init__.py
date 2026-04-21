"""
Evaluation module for the Vanilla S2G pipeline.

Provides metric computation and training callbacks.

Public API
----------
Metrics:
- ``boundary_f1``               — Instance-level boundary micro P/R/F1.
- ``strict_f1``                 — Instance-level strict micro P/R/F1.
- ``ner_boundary_f1``           — Instance-level NER boundary P/R/F1.
- ``ner_typed_f1``              — Instance-level NER typed P/R/F1.
- ``corpus_boundary_f1``        — Corpus-level micro boundary P/R/F1.
- ``corpus_strict_f1``          — Corpus-level micro strict P/R/F1.
- ``corpus_ner_f1``             — Corpus-level micro NER P/R/F1.
- ``compute_metrics``           — Unified entry point (mode-selectable).

Callbacks:
- ``StepTrackingCallback``          — Collator step synchronisation.
- ``GenerateTextSamplesCallback``   — Periodic W&B sample table logging.
- ``PeriodicCheckpointCallback``    — Fixed-interval safety-net checkpoints.

Utilities:
- ``load_run_metadata``             — Recover W&B run ID for resumption.
"""

from .callbacks import (
    GenerateTextSamplesCallback,
    PeriodicCheckpointCallback,
    StepTrackingCallback,
    load_run_metadata,
)
from .metrics import (
    boundary_f1,
    compute_metrics,
    corpus_boundary_f1,
    corpus_ner_f1,
    corpus_strict_f1,
    ner_boundary_f1,
    ner_typed_f1,
    strict_f1,
)

__all__ = [
    # Metrics
    "boundary_f1",
    "strict_f1",
    "ner_boundary_f1",
    "ner_typed_f1",
    "corpus_boundary_f1",
    "corpus_strict_f1",
    "corpus_ner_f1",
    "compute_metrics",
    # Callbacks
    "StepTrackingCallback",
    "GenerateTextSamplesCallback",
    "PeriodicCheckpointCallback",
    "load_run_metadata",
]