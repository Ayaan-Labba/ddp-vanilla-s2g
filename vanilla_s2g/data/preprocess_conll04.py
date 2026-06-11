"""
Pre-processing script for SpERT-format datasets (CoNLL-04).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import yaml

from vanilla_s2g.linearisation import build_sel, organize_by_entity

logger = logging.getLogger(__name__)


def load_label_maps(config_map_path: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not config_map_path:
        return {}, {}
    path = Path(config_map_path)
    if not path.exists():
        logger.warning(f"Config map file {config_map_path} not found. Using raw labels.")
        return {}, {}
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if not config:
        return {}, {}
        
    entities = config.get("entities", {}) or {}
    relations = config.get("relations", {}) or {}
    return {str(k): str(v) for k, v in entities.items()}, {str(k): str(v) for k, v in relations.items()}


def convert_instance(raw: Dict, entity_map: Dict[str, str], relation_map: Dict[str, str]) -> Optional[Dict]:
    if not raw.get("entities"): return None
    
    tokens = raw["tokens"]
    entities = [{
        "text": " ".join(tokens[int(e["start"]):int(e["end"])]),
        "offset": [int(e["start"]), int(e["end"])],
        "type": entity_map.get(e["type"], e["type"])
    } for e in raw["entities"]]

    relations = [{
        "head": entities[int(r["head"])],
        "tail": entities[int(r["tail"])],
        "type": relation_map.get(r["type"], r["type"])
    } for r in raw.get("relations", []) if int(r["head"]) < len(entities) and int(r["tail"]) < len(entities)]

    types = sorted({r["type"] for r in relations})
    
    entity_blocks = organize_by_entity(entities, relations)
    sel = build_sel(entity_blocks, rejected_types=[])

    return {
        "text": " ".join(tokens), "tokens": tokens, "entities": entities, "relations": relations,
        "entity_types": sorted({e["type"] for e in entities}), "rel_types": types,
        "types": types, "sel": sel,
    }


def process_split(
    split_name: str, instances: List[Dict], output_path: Path,
    entity_map: Dict[str, str], relation_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    seen_ent: Set[str] = set()
    seen_rel: Set[str] = set()
    skipped, written = 0, 0
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for raw in instances:
            if inst := convert_instance(raw, entity_map, relation_map):
                fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                seen_ent.update(inst["entity_types"])
                seen_rel.update(inst["rel_types"])
                written += 1
            else: 
                skipped += 1
            
    logger.info(
        "Split: %s → %s (%d written, %d skipped)", 
        split_name, 
        output_path.name, 
        written,
        skipped
    )
    return list(seen_ent), list(seen_rel)


def _write_schema(path: Path, types: List[str]) -> None:
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(unique) + "\n")
    logger.info("Schema: %s (%d types)", path.name, len(unique))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess dataset for S2G fine-tuning.")
    parser.add_argument("--input_file", required=True, help="Path to coll04.json") 
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_map", default="configs/data/conll04.yaml", help="Path to the config YAML for label mapping.")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_map, relation_map = load_label_maps(args.config_map)

    with open(input_file, encoding="utf-8") as fin:
        dataset = json.load(fin)

    all_ent, all_rel = [], []
    split_mapping = {"train": "train.jsonl", "dev": "val.jsonl", "test": "test.jsonl"}
    
    for split_key, out_name in split_mapping.items():
        if split_key in dataset:
            e, r = process_split(split_key, dataset[split_key], output_dir / out_name, entity_map, relation_map)
            if split_key == "train": 
                all_ent, all_rel = e, r

    _write_schema(output_dir / "entity.schema", all_ent)
    _write_schema(output_dir / "relation.schema", all_rel)


if __name__ == "__main__":
    main()
