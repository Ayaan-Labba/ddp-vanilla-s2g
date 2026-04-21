"""
Configuration loader for the S2G pipeline.

Loads a YAML configuration file and merges command-line overrides into a
single flat namespace.  This provides the unified config object consumed
by all training, evaluation, and inference scripts.

Design: CLI arguments take precedence over YAML values, which take
precedence over hard-coded defaults.  This layering enables W&B sweeps
(which inject CLI args) to override any YAML parameter without touching
the config file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[list] = None,
) -> argparse.Namespace:
    """Load configuration from YAML and merge CLI overrides.

    The function performs three steps:

    1. Parse a minimal CLI to identify the ``--config`` path and collect
       all remaining ``--key value`` pairs as overrides.
    2. Load the YAML file into a flat dict.
    3. Apply CLI overrides, casting values to match the YAML type.

    Args:
        config_path: Explicit path to the YAML file (skips CLI parsing
                     for the ``--config`` flag if provided).
        cli_args:    Explicit CLI arg list (defaults to ``sys.argv[1:]``).

    Returns:
        An :class:`argparse.Namespace` with all configuration values.
    """
    # --- Step 1: Parse CLI ---
    parser = argparse.ArgumentParser(
        description="S2G Training / Evaluation",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to the YAML configuration file.",
    )

    # First parse: extract --config and collect unknowns.
    known, unknown = parser.parse_known_args(cli_args)
    yaml_path = config_path or getattr(known, "config", None)

    # --- Step 2: Load YAML ---
    yaml_config: Dict[str, Any] = {}
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", yaml_path)

    # --- Step 3: Parse overrides from unknown args ---
    overrides = _parse_overrides(unknown)

    # Merge: YAML first, then CLI overrides on top.
    merged: Dict[str, Any] = {**yaml_config}
    for key, raw_value in overrides.items():
        if key in merged:
            merged[key] = _cast_value(raw_value, type(merged[key]))
        else:
            merged[key] = _auto_cast(raw_value)

    # Store the config path for reference.
    merged["config_path"] = str(yaml_path) if yaml_path else None

    return argparse.Namespace(**merged)


def _parse_overrides(args: list) -> Dict[str, str]:
    """Parse ``--key value`` pairs from an unknown-args list.

    Supports both ``--key value`` and ``--key=value`` forms.  Boolean
    flags without a value (``--flag``) are set to ``"true"``.
    """
    overrides: Dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                k, v = key.split("=", 1)
                overrides[k] = v
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                overrides[key] = args[i + 1]
                i += 1
            else:
                overrides[key] = "true"
        i += 1
    return overrides


def _cast_value(raw: str, target_type: type) -> Any:
    """Cast a string value to match the target YAML type."""
    if target_type is bool:
        return raw.lower() in ("true", "1", "yes")
    if target_type is type(None):
        if raw.lower() in ("null", "none", ""):
            return None
        # YAML default was null — apply auto-casting (handles comma-
        # separated lists, ints, floats, booleans, and plain strings).
        return _auto_cast(raw)
    if target_type is list:
        # CLI list override: comma-separated, e.g., --gpu_ids 0,1,7
        items = raw.split(",")
        return [_auto_cast_scalar(item.strip()) for item in items if item.strip()]
    try:
        return target_type(raw)
    except (ValueError, TypeError):
        return raw


def _auto_cast(raw: str) -> Any:
    """Best-effort cast when no target type is known."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("null", "none"):
        return None
    # Comma-separated values → list of auto-cast items.
    if "," in raw:
        items = [item.strip() for item in raw.split(",")]
        return [_auto_cast_scalar(item) for item in items if item]
    return _auto_cast_scalar(raw)


def _auto_cast_scalar(raw: str) -> Any:
    """Auto-cast a single scalar value (no commas)."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def load_schema(schema_path: str) -> list:
    """Load a relation schema file (one type per line).

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