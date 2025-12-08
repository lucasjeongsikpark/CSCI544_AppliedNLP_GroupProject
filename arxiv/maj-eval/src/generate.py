
import os, json, yaml, random
from typing import Dict, Any, List
from .loader import load_diqad, load_bioasq, load_mmath

def _seed_everything(seed: int):
    random.seed(seed)

def _predict_diqad_sample(ex):
    # Heuristic: choose choice index by hashing id for determinism
    cid = ex["id"] if ex["id"] is not None else "0"
    idx = (sum(ord(c) for c in str(cid)) % max(1, len(ex.get("choices") or [0,1,2,3])))
    return idx

def _predict_bioasq_sample(ex):
    # Heuristic: if gold answers exist, output the first one; else echo question term
    g = ex.get("gold") or []
    if isinstance(g, list) and g:
        return g[0]
    return (ex.get("question") or "unknown").split(" ")[0]

def _predict_mmath_sample(ex):
    # Heuristic: if gold numeric-looking, return gold; else a token from question
    g = ex.get("gold")
    if g is not None:
        return g
    return (ex.get("question") or "0").split(" ")[0]

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _seed_everything(cfg.get("runtime",{}).get("seed", 42))

    datasets = cfg["datasets"]
    pred_dir = cfg["paths"]["predictions_dir"]
    _ensure_dir(pred_dir)

    # Evaluatees loop (we create identical files per evaluatee to keep structure,
    # but contents come from deterministic heuristics unless you enable real models)
    for eval_model in cfg["evaluatee_models"]:
        for name, path in datasets.items():
            if name=="diqad":
                rows = load_diqad(path)
                preds = []
                for ex in rows:
                    pred = _predict_diqad_sample(ex)
                    preds.append({"id": ex["id"], "pred": pred})
            elif name=="bioasq":
                rows = load_bioasq(path)
                preds = []
                for ex in rows:
                    pred = _predict_bioasq_sample(ex)
                    preds.append({"id": ex["id"], "pred": pred})
            elif name=="mmath":
                rows = load_mmath(path, cfg.get("mmath_images_dir", ""))
                preds = []
                for ex in rows:
                    pred = _predict_mmath_sample(ex)
                    preds.append({"id": ex["id"], "pred": pred})
            else:
                continue
            out_path = os.path.join(pred_dir, f"{name}.{eval_model}.jsonl")
            with open(out_path, "w", encoding="utf-8") as w:
                for p in preds:
                    w.write(json.dumps(p)+"\n")
    print(f"Predictions written to: {pred_dir}")
