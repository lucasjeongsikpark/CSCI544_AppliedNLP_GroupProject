
import os, json, yaml
from typing import Dict, Any, List
from .loader import load_diqad, load_bioasq, load_mmath
from .metrics import accuracy, macro_f1, score_list_em

def _read_preds(path: str) -> Dict[str, Any]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[str(row["id"])] = row["pred"]
    return out

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = cfg["datasets"]
    pred_dir = cfg["paths"]["predictions_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    _ensure_dir(metrics_dir)

    aggregate = {}

    for name, path in datasets.items():
        if name=="diqad":
            rows = load_diqad(path)
            golds = [ex.get("gold") for ex in rows]
            results = {}
            for model in cfg["evaluatee_models"]:
                rp = os.path.join(pred_dir, f"{name}.{model}.jsonl")
                if not os.path.exists(rp): 
                    continue
                preds_map = _read_preds(rp)
                preds = [preds_map.get(str(ex["id"])) for ex in rows]
                # Force string normalization to avoid mixed-type issues
                gnorm = [str(g) for g in golds]
                pnorm = [str(p) for p in preds]
                acc = accuracy(gnorm, pnorm)
                mf1 = macro_f1(gnorm, pnorm)
                results[model] = {"accuracy": acc, "macro_f1": mf1}
            aggregate[name] = results
        elif name=="bioasq":
            rows = load_bioasq(path)
            golds = [ex.get("gold") for ex in rows]
            results = {}
            for model in cfg["evaluatee_models"]:
                rp = os.path.join(pred_dir, f"{name}.{model}.jsonl")
                if not os.path.exists(rp): 
                    continue
                preds_map = _read_preds(rp)
                preds = [preds_map.get(str(ex["id"])) for ex in rows]
                em = score_list_em(golds, preds)
                results[model] = {"exact_match": em}
            aggregate[name] = results
        elif name=="mmath":
            rows = load_mmath(path, cfg.get("mmath_images_dir",""))
            golds = [ex.get("gold") for ex in rows]
            results = {}
            for model in cfg["evaluatee_models"]:
                rp = os.path.join(pred_dir, f"{name}.{model}.jsonl")
                if not os.path.exists(rp): 
                    continue
                preds_map = _read_preds(rp)
                preds = [preds_map.get(str(ex["id"])) for ex in rows]
                em = score_list_em(golds, preds)
                results[model] = {"exact_match": em}
            aggregate[name] = results

    out_json = os.path.join(metrics_dir, "aggregate_results.json")
    with open(out_json, "w", encoding="utf-8") as w:
        json.dump(aggregate, w, indent=2)
    print(f"Wrote metrics -> {out_json}")
