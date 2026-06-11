"""
Pre-processing script for the SciERC dataset.
Supports standard sentence-level and broader document-level parsing.
Strictly maps document-absolute indexing formats to target requirements.
"""
from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
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


def _build_instance(tokens: List[str], entities: List[Dict], relations: List[Dict]) -> Optional[Dict]:
    if not entities: 
        return None
    
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


def convert_document(doc: Dict, entity_map: Dict[str, str], relation_map: Dict[str, str]) -> List[Dict]:
    sentences, ner_per_sent, rel_per_sent = doc["sentences"], doc.get("ner", []), doc.get("relations", [])
    instances = []

    # Map sequential offsets (SciERC from PURE indexes across the entire doc)
    sent_offsets = [0]
    for s in sentences[:-1]:
        sent_offsets.append(sent_offsets[-1] + len(s))

    for i, tokens in enumerate(sentences):
        ner = ner_per_sent[i] if i < len(ner_per_sent) else []
        rel = rel_per_sent[i] if i < len(rel_per_sent) else []
        n_tok = len(tokens)
        off = sent_offsets[i]
        
        entities, span_idx = [], {}
        for s, e, t in ner:
            s_int, e_int = int(s), int(e)
            
            # Deterministic absolute to relative transformation
            s_rel, e_rel = s_int - off, e_int - off

            if 0 <= s_rel <= e_rel < n_tok and (s_rel, e_rel + 1) not in span_idx:
                span_idx[(s_rel, e_rel + 1)] = len(entities)
                mapped_type = entity_map.get(t, t)
                entities.append({
                    "text": " ".join(tokens[s_rel:e_rel+1]), 
                    "offset": [s_rel, e_rel+1], 
                    "type": mapped_type
                })

        relations = []
        for r in rel:
            h_s, h_e, t_s, t_e = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            
            # Deterministic absolute to relative transformation
            h_s, h_e, t_s, t_e = h_s - off, h_e - off, t_s - off, t_e - off
            
            h_key, t_key = (h_s, h_e + 1), (t_s, t_e + 1)
            if h_key in span_idx and t_key in span_idx:
                mapped_rel_type = relation_map.get(r[4], r[4])
                relations.append({
                    "head": entities[span_idx[h_key]], 
                    "tail": entities[span_idx[t_key]], 
                    "type": mapped_rel_type
                })

        if inst := _build_instance(tokens, entities, relations): 
            instances.append(inst)

    return instances


def convert_document_level(doc: Dict, entity_map: Dict[str, str], relation_map: Dict[str, str]) -> Optional[Dict]:
    doc_tokens, sent_offsets = [], []
    for sent in doc["sentences"]:
        sent_offsets.append(len(doc_tokens))
        doc_tokens.extend(sent)

    entities, span_to_ent = [], {}
    for i, ner in enumerate(doc.get("ner", [])):
        for s, e, t in ner:
            s_abs, e_abs = int(s), int(e)

            key = (s_abs, e_abs + 1)
            if key not in span_to_ent and 0 <= s_abs <= e_abs < len(doc_tokens):
                mapped_type = entity_map.get(t, t)
                ent = {"text": " ".join(doc_tokens[key[0]:key[1]]), "offset": list(key), "type": mapped_type}
                span_to_ent[key] = ent
                entities.append(ent)

    relations = []
    for i, rel in enumerate(doc.get("relations", [])):
        for r in rel:
            h_s_abs, h_e_abs, t_s_abs, t_e_abs = int(r[0]), int(r[1]), int(r[2]), int(r[3])

            h_key, t_key = (h_s_abs, h_e_abs + 1), (t_s_abs, t_e_abs + 1)
            if h_key in span_to_ent and t_key in span_to_ent:
                mapped_rel_type = relation_map.get(r[4], r[4])
                relations.append({
                    "head": span_to_ent[h_key], 
                    "tail": span_to_ent[t_key], 
                    "type": mapped_rel_type
                })

    return _build_instance(doc_tokens, entities, relations)


def process_split(
    input_path: Path, output_path: Path, document_level: bool,
    entity_map: Dict[str, str], relation_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    seen_ent: Set[str] = set()
    seen_rel: Set[str] = set()
    written = 0
    
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in filter(None, (ln.strip() for ln in fin)):
            doc = json.loads(line)
            
            if document_level:
                if d_inst := convert_document_level(doc, entity_map, relation_map):
                    fout.write(json.dumps(d_inst, ensure_ascii=False) + "\n")
                    seen_ent.update(d_inst["entity_types"])
                    seen_rel.update(d_inst["rel_types"])
                    written += 1
            else:
                for inst in convert_document(doc, entity_map, relation_map):
                    fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                    seen_ent.update(inst["entity_types"])
                    seen_rel.update(inst["rel_types"])
                    written += 1
        
    logger.info("%s → %s (%d instances written)", input_path.name, output_path.name, written)
    return list(seen_ent), list(seen_rel)


def _write_schema(path: Path, types: List[str]) -> None:
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f: 
        f.write("\n".join(unique) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess SciERC for S2G fine-tuning.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--document_level", action="store_true")
    parser.add_argument("--config_map", default="configs/data/scierc.yaml", help="Path to the config YAML for label mapping.")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_map, relation_map = load_label_maps(args.config_map)

    all_ent, all_rel = [], []
    for in_name, out_name in {"train.json": "train.jsonl", "dev.json": "val.jsonl", "test.json": "test.jsonl"}.items():
        if (in_path := input_dir / in_name).exists():
            e, r = process_split(
                in_path, 
                output_dir / out_name, 
                args.document_level,
                entity_map,
                relation_map
            )
            if in_name == "train.json": 
                all_ent, all_rel = e, r

    _write_schema(output_dir / "entity.schema", all_ent)
    _write_schema(output_dir / "relation.schema", all_rel)


if __name__ == "__main__":
    main()
