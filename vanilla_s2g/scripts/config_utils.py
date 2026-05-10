"""
Configuration loader for the S2G pipeline (OmegaConf-based).

Defines a fully typed, nested schema (``S2GConfig`` and its subsection
dataclasses) and merges three layers in resolution order:

    1. Dataclass defaults  — the schema, source of truth for fields.
    2. YAML overlay        — per-experiment values (configs/*.yaml).
    3. CLI dotlist         — per-run overrides (key=value, dotted for
                              nested fields, e.g. ``optimizer.lr=3e-5``).

Because the schema is enforced in *struct mode*, unknown keys (typos,
stale fields) raise a clear error at load time rather than silently
shadowing the real value.  Types are enforced too: ``optimizer.lr=abc``
fails immediately instead of crashing later inside the optimiser.

CLI dotlist syntax is exactly the format produced by W&B sweeps via the
``${args_no_hyphens}`` expansion, so sweeps work without translation.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ===================================================================== #
#                       NESTED CONFIG SCHEMA                             #
# ===================================================================== #
#
# Each subsection is a dataclass with typed fields and sensible defaults.
# Group boundaries follow the natural "concerns" of the pipeline; a flat
# key like ``train_batch_size`` becomes ``train.batch_size``, and a key
# like ``eval_beams`` becomes ``generation.num_beams``.
#
# Optional fields use ``Optional[...] = None``.  This signals to OmegaConf
# that ``null`` in YAML or ``key=null`` on the CLI is acceptable.


@dataclass
class ModelConfig:
    """Backbone selection and (optional) fine-tuning starting point."""
    name: str = "google/flan-t5-base"
    pretrained_checkpoint: Optional[str] = None  # Set during fine-tuning


@dataclass
class TokenizationConfig:
    """Encoder / decoder length budgets in subword tokens."""
    max_source_length: int = 400
    max_target_length: int = 200


@dataclass
class OptimizerConfig:
    """AdamW optimiser hyperparameters."""
    lr: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning-rate schedule."""
    type: str = "cosine"
    warmup_steps: int = 15_000          # ~10% of max training steps


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""
    max_steps: int = 150_000            # ~6 epochs on ~784K examples with effective batch_size=32
    batch_size: int = 8                 # Per-device train batch size
    gradient_acc_steps: int = 4
    gradient_clip_value: float = 10.0
    precision: str = "bf16"             # "16", "bf16", or "32"
    seed: int = 0


@dataclass
class ValidationConfig:
    """Validation loop and early stopping."""
    check_interval: int = 6125
    percent_check: float = 0.1
    batch_size: int = 32               # Per-device eval batch size
    early_stopping_patience: int = 12
    early_stopping_metric: str = "val_f1"


@dataclass
class GenerationConfig:
    """Beam-search settings used at validation and evaluation."""
    num_beams: int = 3
    length_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False
    constraint_decoding: bool = False


@dataclass
class SSIConfig:
    """Schema-Sensitive-Input dynamic construction."""
    positive_rate: float = 0.9
    negative_rate: float = 0.1
    negative_max_start: int = 1                # k(0)
    negative_max_end: int = 20                 # k(T)
    max_types_in_prompt: Optional[int] = None  # null = use k(t) schedule
    random_prompt: bool = False
    random_sel: bool = False


@dataclass
class TypedSELConfig:
    """Typed SEL generation (fine-tuning only)."""
    enabled: bool = False
    entity_types: List[str] = field(default_factory=list)


@dataclass
class CheckpointConfig:
    """Checkpointing and resumption."""
    save_top_k: int = 3
    metric: str = "val_f1"
    mode: str = "max"
    save_last: bool = True
    every_n_steps: int = 500
    resume_from: Optional[str] = None


@dataclass
class CallbacksConfig:
    """Custom-callback intervals."""
    sample_generation_interval: int = 12_250


@dataclass
class WandbConfig:
    """Weights & Biases run metadata."""
    project: str = "ddp-vanilla-s2g"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class DataConfig:
    """Data and output paths."""
    data_dir: Optional[str] = None
    schema_file: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class HardwareConfig:
    """GPU selection and dataloader workers."""
    num_workers: int = 0
    gpu_ids: Optional[List[int]] = None


@dataclass
class EvaluationConfig:
    """Final-evaluation settings (consumed by ``evaluate.py``).

    Only ``evaluate.py`` reads this section; ``pretrain.py`` and
    ``finetune.py`` ignore it.  Fields kept here are those that have no
    natural home in the existing training-oriented sections.
    """
    split: str = "test"                 # "val" or "test"
    mode: str = "boundary"              # "boundary" or "strict" — metric mode


@dataclass
class S2GConfig:
    """Top-level config, aggregating every nested subsection."""
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    ssi: SSIConfig = field(default_factory=SSIConfig)
    typed_sel: TypedSELConfig = field(default_factory=TypedSELConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Provenance: filled in by load_config().
    config_path: Optional[str] = None


# ===================================================================== #
#                             LOADER                                     #
# ===================================================================== #


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
) -> DictConfig:
    """Build a typed, nested config from defaults, YAML, and CLI.

    Resolution order (later layers override earlier):

        1. Schema defaults (the dataclass tree).
        2. YAML at ``config_path`` (or ``--config <path>`` parsed from
           ``cli_args``).
        3. CLI dotlist overrides — bare ``key=value`` pairs, dotted
           for nested fields (e.g. ``optimizer.lr=3e-5``).

    Args:
        config_path: Explicit path to the YAML overlay.  If ``None``,
                     the loader looks for ``--config <path>`` in
                     ``cli_args``.
        cli_args:    Override list, defaults to ``sys.argv[1:]``.

    Returns:
        An OmegaConf ``DictConfig`` with structured-mode enforcement.
        Access nested fields by attribute, e.g. ``cfg.optimizer.lr``.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        ValueError: if a CLI arg is in legacy ``--key value`` form or
                    is not a valid ``key=value`` dotlist entry.
    """
    if cli_args is None:
        cli_args = sys.argv[1:]

    # ---- Step 1: extract --config <path>, leave the rest as dotlist ----
    yaml_path, remaining = _extract_config_flag(cli_args, config_path)
    _validate_dotlist(remaining)

    # ---- Step 2: typed schema as the strict base ----
    cfg = OmegaConf.structured(S2GConfig)

    # ---- Step 3: YAML overlay ----
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        file_cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, file_cfg)
        logger.info("Loaded config from %s", path)
    else:
        logger.warning(
            "No --config path provided; using schema defaults only. "
            "Pass --config <path> to load a YAML overlay."
        )

    # ---- Step 4: CLI dotlist overrides ----
    if remaining:
        cli_cfg = OmegaConf.from_dotlist(remaining)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        logger.info("Applied %d CLI override(s).", len(remaining))

    # Stash provenance for downstream logging / metadata.
    cfg.config_path = str(yaml_path) if yaml_path else None
    return cfg


def _extract_config_flag(
    cli_args: List[str],
    explicit_path: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    """Strip ``--config <path>`` (or ``--config=<path>``) from cli_args.

    Returns ``(yaml_path, remaining_args)``.  An ``explicit_path``
    argument always wins over any ``--config`` flag found in
    ``cli_args``, but the flag itself is still removed from the
    remaining args so it doesn't reach the dotlist parser.
    """
    yaml_path: Optional[str] = explicit_path
    remaining: List[str] = []
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg == "--config":
            if i + 1 >= len(cli_args):
                raise ValueError("--config flag requires a path argument.")
            if explicit_path is None:
                yaml_path = cli_args[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            if explicit_path is None:
                yaml_path = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return yaml_path, remaining


def _validate_dotlist(args: List[str]) -> None:
    """Ensure remaining args are in OmegaConf dotlist form (key=value).

    Catches the common mistake of typing ``--lr 3e-5`` instead of
    ``optimizer.lr=3e-5`` and emits a helpful error rather than letting
    OmegaConf raise an opaque parse failure later.
    """
    for arg in args:
        if arg.startswith("-"):
            raise ValueError(
                f"Unrecognised CLI flag: '{arg}'.  Overrides must be "
                "in dotlist form, e.g. 'optimizer.lr=3e-5' (no leading "
                "dashes).  The only flag accepted is '--config <path>'."
            )
        if "=" not in arg:
            raise ValueError(
                f"Malformed override: '{arg}'.  Expected 'key=value' "
                "(use dotted keys for nested fields, e.g. "
                "'optimizer.lr=3e-5')."
            )


# ===================================================================== #
#                        RELATION SCHEMA                                 #
# ===================================================================== #


def load_schema(schema_path: str) -> List[str]:
    """Load a relation schema file (one type per line).

    Preserved verbatim from the previous loader so that ``evaluate.py``
    and ``inference.py`` keep working without modification.

    Args:
        schema_path: Path to the ``relation.schema`` file.

    Returns:
        List of relation-type strings.
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]