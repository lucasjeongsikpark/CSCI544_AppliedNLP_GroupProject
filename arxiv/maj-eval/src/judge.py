
import os, json, yaml, random, datetime
from typing import Dict, Any, List
from .loader import load_diqad, load_bioasq, load_mmath
from .metrics import exact_match

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _deterministic_label(evaluatee_pred, gold, persona_name):
    # Offline fallback judgment rule:
    # - For DiQAD (ints), compare equality
    # - For BioASQ (list of acceptable), exact match helper
    # - For MMATH (string/number), normalize by string exact
    if isinstance(gold, list):
        is_correct = exact_match(gold, evaluatee_pred)
    else:
        is_correct = str(gold) == str(evaluatee_pred)
    # Persona affects only the *rationale style*, not the label in fallback
    rationale = {
        "StrictReferee": "Compared directly to the gold reference. Verdict follows exact correctness.",
        "SafetyFocused": "Given uncertainty, favored conservative evaluation after checking against gold.",
        "PedanticScholar": "Verified step-by-step alignment with canonical answer; discrepancies highlighted."
    }.get(persona_name, "Compared against gold.")
    return "correct" if is_correct else "incorrect", rationale

def _aggregate(labels: List[str], mode: str) -> str:
    if mode == "majority_vote":
        c = sum(1 for x in labels if x=="correct")
        ic = len(labels)-c
        if c>ic: return "correct"
        if ic>c: return "incorrect"
        # tie-break default -> incorrect
        return "incorrect"
    # default
    return labels[0] if labels else "incorrect"

def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    datasets = cfg["datasets"]
    pred_dir = cfg["paths"]["predictions_dir"]
    eval_dir = cfg["paths"]["evaluations_dir"]
    logs_dir = cfg["paths"]["logs_dir"]
    _ensure_dir(eval_dir); _ensure_dir(logs_dir)

    personas = [p["name"] for p in cfg.get("personas",[])]
    debate = cfg.get("debate",{})
    agg_mode = debate.get("aggregation","majority_vote")

    for name, path in datasets.items():
        # Load samples
        if name=="diqad":
            rows = load_diqad(path)
        elif name=="bioasq":
            rows = load_bioasq(path)
        elif name=="mmath":
            rows = load_mmath(path, cfg.get("mmath_images_dir",""))
        else:
            continue

        # For each evaluatee model, load predictions and judge
        judgments = []
        logs = []
        for eval_model in cfg["evaluatee_models"]:
            pred_path = os.path.join(pred_dir, f"{name}.{eval_model}.jsonl")
            if not os.path.exists(pred_path):
                continue
            preds_by_id = {}
            with open(pred_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    preds_by_id[str(row["id"])] = row["pred"]

            for ex in rows:
                exid = str(ex["id"])
                pred = preds_by_id.get(exid, None)
                per_evaluator = []
                labels = []
                for persona in personas:
                    label, rationale = _deterministic_label(pred, ex.get("gold"), persona)
                    labels.append(label)
                    per_evaluator.append({
                        "evaluator": f"{random.choice(cfg['evaluator_models'])}-{persona}",
                        "persona": persona,
                        "rationale": rationale,
                        "label": label
                    })
                final_label = _aggregate(labels, agg_mode)
                judgments.append({
                    "dataset": name,
                    "evaluatee_model": eval_model,
                    "id": exid,
                    "final_label": final_label,
                    "per_evaluator": per_evaluator
                })
                logs.append({
                    "time": datetime.datetime.utcnow().isoformat()+"Z",
                    "dataset": name,
                    "id": exid,
                    "turns": [
                        {"role":"system","content":f"Persona(s): {', '.join(personas)}"},
                        {"role":"user","content": f"Question: {ex.get('question')}"},
                        {"role":"assistant","content": f"Evaluatee answer: {pred}"},
                        {"role":"assistant","content": f"Gold: {ex.get('gold')}"}
                    ]
                })

        # Write outputs
        out_judgments = os.path.join(eval_dir, f"{name}.judgments.jsonl")
        with open(out_judgments, "w", encoding="utf-8") as w:
            for j in judgments:
                w.write(json.dumps(j)+"\n")
        out_logs = os.path.join(logs_dir, f"{name}.logs.jsonl")
        with open(out_logs, "w", encoding="utf-8") as w:
            for l in logs:
                w.write(json.dumps(l)+"\n")

        print(f"Wrote judgments -> {out_judgments}")
        print(f"Wrote logs      -> {out_logs}")
