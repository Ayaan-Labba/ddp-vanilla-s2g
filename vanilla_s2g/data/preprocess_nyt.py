"""
Pre-processing script for raw NYT dataset (from the ReLiK repo) into S2G fine-tuning format.
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


def find_sublist_index(sub: List[str], full: List[str]) -> int:
    """Finds the starting index of a sublist within a full list."""
    n = len(sub)
    for i in range(len(full) - n + 1):
        if full[i:i+n] == sub:
            return i
    return -1


def convert_instance(raw: Dict, entity_map: Dict[str, str], relation_map: Dict[str, str]) -> Optional[Dict]:
    if not raw.get("entityMentions"): 
        return None
    
    sent_text = raw.get("sentText", "")
    tokens = sent_text.split()
    
    entities = []
    entity_text_to_idx = {}

    for raw_ent in raw["entityMentions"]:
        e_text = raw_ent["text"]
        e_tokens = e_text.split()
        
        start_idx = find_sublist_index(e_tokens, tokens)
        
        if start_idx != -1:
            end_idx = start_idx + len(e_tokens)
            mapped_type = entity_map.get(raw_ent["label"], raw_ent["label"])
            entities.append({
                "text": " ".join(tokens[start_idx:end_idx]),
                "offset": [start_idx, end_idx],
                "type": mapped_type
            })
            entity_text_to_idx[e_text] = len(entities) - 1

    if not entities:
        return None

    relations = []
    for raw_rel in raw.get("relationMentions", []):
        head_text = raw_rel.get("em1Text")
        tail_text = raw_rel.get("em2Text")
        
        if head_text in entity_text_to_idx and tail_text in entity_text_to_idx:
            head_idx = entity_text_to_idx[head_text]
            tail_idx = entity_text_to_idx[tail_text]
            raw_rel_type = raw_rel["label"].split("/")[-1]
            mapped_rel_type = relation_map.get(raw_rel_type, raw_rel_type)
            relations.append({
                "head": entities[head_idx],
                "tail": entities[tail_idx],
                "type": mapped_rel_type
            })

    types = sorted({r["type"] for r in relations})
    
    entity_blocks = organize_by_entity(entities, relations)
    sel = build_sel(entity_blocks, rejected_types=[])

    return {
        "text": " ".join(tokens), 
        "tokens": tokens, 
        "entities": entities, 
        "relations": relations,
        "entity_types": sorted({e["type"] for e in entities}), 
        "rel_types": types,
        "types": types,
        "sel": sel,
    }


def process_split(
    input_path: Path, output_path: Path,
    entity_map: Dict[str, str], relation_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    seen_ent: Set[str] = set()
    seen_rel: Set[str] = set()
    skipped, written = 0, 0
    
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            raw = json.loads(line)
            
            if inst := convert_instance(raw, entity_map, relation_map):
                fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                seen_ent.update(inst["entity_types"])
                seen_rel.update(inst["rel_types"])
                written += 1
            else: 
                skipped += 1
            
    logger.info(
        "%s → %s (%d written, %d skipped)", 
        input_path.name, 
        output_path.name, 
        written,
        skipped
    )
    return list(seen_ent), list(seen_rel)


def _write_schema(path: Path, types: List[str]) -> None:
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f: 
        f.write("\n".join(unique) + "\n")
    logger.info("Schema: %s (%d types)", path.name, len(unique))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess dataset for S2G fine-tuning.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_map", default="configs/data/nyt.yaml", help="Path to the config YAML for label mapping.")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_map, relation_map = load_label_maps(args.config_map)

    all_ent, all_rel = [], []
    file_map = {
        "raw_train.json": "train.jsonl", 
        "raw_valid.json": "val.jsonl", 
        "raw_test.json": "test.jsonl"
    }
    
    for in_name, out_name in file_map.items():
        if (in_path := input_dir / in_name).exists():
            e, r = process_split(in_path, output_dir / out_name, entity_map, relation_map)
            if in_name == "raw_train.json": 
                all_ent, all_rel = e, r

    _write_schema(output_dir / "entity.schema", all_ent)
    _write_schema(output_dir / "relation.schema", all_rel)


if __name__ == "__main__":
    main()
