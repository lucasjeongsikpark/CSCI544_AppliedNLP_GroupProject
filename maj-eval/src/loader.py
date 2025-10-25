
import json, os
from typing import List, Dict, Any

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def load_diqad(path: str) -> List[Dict[str, Any]]:
    data = _read_jsonl(path)
    # Normalize: expect fields like {"id", "question", "choices", "gold"} or {"label"} variants
    out = []
    for ex in data:
        gold = ex.get("gold", ex.get("label"))
        out.append({
            "id": ex.get("id"),
            "question": ex.get("question"),
            "choices": ex.get("choices"),
            "gold": gold,
            "meta": {k:v for k,v in ex.items() if k not in ("id","question","choices","gold","label")}
        })
    return out

def load_bioasq(path: str) -> List[Dict[str, Any]]:
    data = _read_jsonl(path)
    out = []
    for ex in data:
        gold = ex.get("gold_exact") or ex.get("gold") or []
        out.append({
            "id": ex.get("id"),
            "question": ex.get("question") or ex.get("body"),
            "context": ex.get("context"),
            "gold": gold, # list of acceptable answers
            "meta": {k:v for k,v in ex.items() if k not in ("id","question","body","context","gold","gold_exact")}
        })
    return out

def load_mmath(path: str, images_dir: str) -> List[Dict[str, Any]]:
    data = _read_jsonl(path)
    out = []
    for ex in data:
        img = ex.get("image") or ex.get("image_path")
        if img and not os.path.isabs(img):
            img = os.path.join(images_dir, img)
        out.append({
            "id": ex.get("id"),
            "question": ex.get("question"),
            "image_path": img,
            "gold": ex.get("solution") or ex.get("gold"),
            "meta": {k:v for k,v in ex.items() if k not in ("id","question","image","image_path","gold","solution")}
        })
    return out
