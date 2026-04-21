"""
Evaluation metrics for the S2G model.

Provides micro Precision, Recall, and F1 at two granularity levels:

1. **Boundary F1** — Matches ``(head_text, relation_type, tail_text)``
   triplets without entity typing.  Used during pre-training (REBEL has
   no entity type annotations).

2. **Strict F1** — Matches ``(head_text, head_type, relation_type,
   tail_text, tail_type)`` quintuples requiring exact entity-type match.
   Used during fine-tuning on benchmarks that provide entity types.

3. **NER F1** — Entity-only evaluation matching ``(entity_text,
   entity_type)`` pairs.  Boundary variant ignores type.

All functions accept lists of predicted and gold items and return a
dictionary of metric values.  A unified ``compute_metrics`` entry point
selects the appropriate level based on a ``mode`` argument.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Triplet = Tuple[str, str, str]  # (head, rel_type, tail)
Quintuple = Tuple[str, str, str, str, str]  # (head, head_type, rel_type, tail, tail_type)
EntityMention = Tuple[str, str]  # (text, type)


# ===================================================================== #
#                      CORE P / R / F1                                   #
# ===================================================================== #


def _prf(
    predicted: Set,
    gold: Set,
) -> Dict[str, float]:
    """Compute micro Precision, Recall, and F1 between two sets.

    Args:
        predicted: Set of predicted items.
        gold:      Set of gold items.

    Returns:
        ``{"precision": ..., "recall": ..., "f1": ...}``
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


# ===================================================================== #
#                     BOUNDARY F1 (PRE-TRAINING)                         #
# ===================================================================== #


def boundary_f1(
    predicted_triplets: List[Triplet],
    gold_triplets: List[Triplet],
) -> Dict[str, float]:
    """Compute boundary micro P/R/F1 over (head, rel_type, tail) triplets.

    Entity types are ignored — only the surface text spans and the
    relation type must match exactly.

    Args:
        predicted_triplets: List of ``(head, rel_type, tail)`` from the model.
        gold_triplets:      List of ``(head, rel_type, tail)`` from the ground truth.

    Returns:
        ``{"boundary_precision": ..., "boundary_recall": ..., "boundary_f1": ...}``
    """
    pred_set = set(predicted_triplets)
    gold_set = set(gold_triplets)
    raw = _prf(pred_set, gold_set)
    return {
        "boundary_precision": raw["precision"],
        "boundary_recall": raw["recall"],
        "boundary_f1": raw["f1"],
    }


# ===================================================================== #
#                     STRICT F1 (FINE-TUNING)                            #
# ===================================================================== #


def strict_f1(
    predicted_quintuples: List[Quintuple],
    gold_quintuples: List[Quintuple],
) -> Dict[str, float]:
    """Compute strict micro P/R/F1 requiring exact entity-type match.

    Each item is ``(head_text, head_type, rel_type, tail_text, tail_type)``.

    Args:
        predicted_quintuples: Predicted quintuples.
        gold_quintuples:      Gold quintuples.

    Returns:
        ``{"strict_precision": ..., "strict_recall": ..., "strict_f1": ...}``
    """
    pred_set = set(predicted_quintuples)
    gold_set = set(gold_quintuples)
    raw = _prf(pred_set, gold_set)
    return {
        "strict_precision": raw["precision"],
        "strict_recall": raw["recall"],
        "strict_f1": raw["f1"],
    }


# ===================================================================== #
#                          NER F1                                        #
# ===================================================================== #


def ner_boundary_f1(
    predicted_entities: List[str],
    gold_entities: List[str],
) -> Dict[str, float]:
    """Compute NER boundary micro P/R/F1 (entity text only, no type).

    Args:
        predicted_entities: List of predicted entity text spans.
        gold_entities:      List of gold entity text spans.

    Returns:
        ``{"ner_boundary_precision": ..., "ner_boundary_recall": ..., "ner_boundary_f1": ...}``
    """
    pred_set = set(predicted_entities)
    gold_set = set(gold_entities)
    raw = _prf(pred_set, gold_set)
    return {
        "ner_boundary_precision": raw["precision"],
        "ner_boundary_recall": raw["recall"],
        "ner_boundary_f1": raw["f1"],
    }


def ner_typed_f1(
    predicted_entities: List[EntityMention],
    gold_entities: List[EntityMention],
) -> Dict[str, float]:
    """Compute NER typed micro P/R/F1 (entity text + type must match).

    Args:
        predicted_entities: List of ``(text, type)`` tuples.
        gold_entities:      List of ``(text, type)`` tuples.

    Returns:
        ``{"ner_typed_precision": ..., "ner_typed_recall": ..., "ner_typed_f1": ...}``
    """
    pred_set = set(predicted_entities)
    gold_set = set(gold_entities)
    raw = _prf(pred_set, gold_set)
    return {
        "ner_typed_precision": raw["precision"],
        "ner_typed_recall": raw["recall"],
        "ner_typed_f1": raw["f1"],
    }


# ===================================================================== #
#                     BATCH-LEVEL AGGREGATION                            #
# ===================================================================== #


def aggregate_batch_metrics(
    per_instance_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute the mean of each metric across a batch of instances.

    Args:
        per_instance_metrics: List of metric dicts (one per instance).

    Returns:
        A single dict with the mean value for each metric key.
    """
    if not per_instance_metrics:
        return {}

    keys = per_instance_metrics[0].keys()
    aggregated: Dict[str, float] = {}
    n = len(per_instance_metrics)

    for key in keys:
        aggregated[key] = sum(m[key] for m in per_instance_metrics) / n

    return aggregated


# ===================================================================== #
#              CORPUS-LEVEL MICRO METRICS (PREFERRED)                    #
# ===================================================================== #


def corpus_boundary_f1(
    all_predicted: List[List[Triplet]],
    all_gold: List[List[Triplet]],
) -> Dict[str, float]:
    """Compute corpus-level micro boundary F1 across all instances.

    Unlike instance-level averaging (macro), this pools all predictions
    and gold items into single sets and computes micro P/R/F1.  This is
    the standard evaluation protocol for relation extraction and is used
    for early stopping and final reporting.

    Args:
        all_predicted: List of predicted triplet lists (one per instance).
        all_gold:      List of gold triplet lists (one per instance).

    Returns:
        ``{"boundary_precision": ..., "boundary_recall": ..., "boundary_f1": ...}``
    """
    pred_set: Set[Tuple[int, str, str, str]] = set()
    gold_set: Set[Tuple[int, str, str, str]] = set()

    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for t in preds:
            pred_set.add((idx, t[0], t[1], t[2]))
        for t in golds:
            gold_set.add((idx, t[0], t[1], t[2]))

    raw = _prf(pred_set, gold_set)
    return {
        "boundary_precision": raw["precision"],
        "boundary_recall": raw["recall"],
        "boundary_f1": raw["f1"],
    }


def corpus_strict_f1(
    all_predicted: List[List[Quintuple]],
    all_gold: List[List[Quintuple]],
) -> Dict[str, float]:
    """Compute corpus-level micro strict F1 across all instances.

    Args:
        all_predicted: List of predicted quintuple lists (one per instance).
        all_gold:      List of gold quintuple lists (one per instance).

    Returns:
        ``{"strict_precision": ..., "strict_recall": ..., "strict_f1": ...}``
    """
    pred_set: Set[Tuple[int, str, str, str, str, str]] = set()
    gold_set: Set[Tuple[int, str, str, str, str, str]] = set()

    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for q in preds:
            pred_set.add((idx, q[0], q[1], q[2], q[3], q[4]))
        for q in golds:
            gold_set.add((idx, q[0], q[1], q[2], q[3], q[4]))

    raw = _prf(pred_set, gold_set)
    return {
        "strict_precision": raw["precision"],
        "strict_recall": raw["recall"],
        "strict_f1": raw["f1"],
    }


def corpus_ner_f1(
    all_predicted: List[List[str]],
    all_gold: List[List[str]],
    typed: bool = False,
) -> Dict[str, float]:
    """Compute corpus-level micro NER F1 across all instances.

    Args:
        all_predicted: List of predicted entity lists (one per instance).
                       Each entity is a string (boundary) or ``(text, type)`` (typed).
        all_gold:      List of gold entity lists.
        typed:         If ``True``, entities are ``(text, type)`` tuples.

    Returns:
        NER metric dict.
    """
    pred_set: set = set()
    gold_set: set = set()

    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for e in preds:
            pred_set.add((idx, e) if not typed else (idx, e[0], e[1]))
        for e in golds:
            gold_set.add((idx, e) if not typed else (idx, e[0], e[1]))

    raw = _prf(pred_set, gold_set)
    prefix = "ner_typed" if typed else "ner_boundary"
    return {
        f"{prefix}_precision": raw["precision"],
        f"{prefix}_recall": raw["recall"],
        f"{prefix}_f1": raw["f1"],
    }


# ===================================================================== #
#                     UNIFIED INTERFACE                                  #
# ===================================================================== #


def compute_metrics(
    all_predicted_triplets: List[List[Triplet]],
    all_gold_triplets: List[List[Triplet]],
    all_predicted_entities: Optional[List[List[str]]] = None,
    all_gold_entities: Optional[List[List[str]]] = None,
    mode: str = "boundary",
) -> Dict[str, float]:
    """Unified metric computation entry point.

    Args:
        all_predicted_triplets: Predicted triplet lists per instance.
        all_gold_triplets:      Gold triplet lists per instance.
        all_predicted_entities: Predicted entity text lists (for NER metrics).
        all_gold_entities:      Gold entity text lists.
        mode:                   ``"boundary"`` (pre-training) or ``"strict"``
                                (fine-tuning).  When ``"strict"``, triplets
                                are expected to be quintuples.

    Returns:
        Combined metric dict.
    """
    metrics: Dict[str, float] = {}

    if mode == "boundary":
        metrics.update(corpus_boundary_f1(all_predicted_triplets, all_gold_triplets))
    elif mode == "strict":
        # In strict mode, triplets are actually quintuples.
        metrics.update(corpus_strict_f1(all_predicted_triplets, all_gold_triplets))  # type: ignore
        # Also compute boundary F1 for the averaged metric.
        boundary_triplets_pred = [
            [(q[0], q[2], q[3]) for q in inst] for inst in all_predicted_triplets
        ]
        boundary_triplets_gold = [
            [(q[0], q[2], q[3]) for q in inst] for inst in all_gold_triplets
        ]
        boundary = corpus_boundary_f1(boundary_triplets_pred, boundary_triplets_gold)
        metrics["boundary_precision"] = boundary["boundary_precision"]
        metrics["boundary_recall"] = boundary["boundary_recall"]
        metrics["boundary_f1"] = boundary["boundary_f1"]
        # Average F1 for early stopping in fine-tuning.
        metrics["avg_f1"] = (metrics["strict_f1"] + metrics["boundary_f1"]) / 2
    else:
        raise ValueError(f"Unknown metrics mode: {mode!r}. Use 'boundary' or 'strict'.")

    # NER metrics (optional).
    if all_predicted_entities is not None and all_gold_entities is not None:
        metrics.update(corpus_ner_f1(all_predicted_entities, all_gold_entities))

    return metrics