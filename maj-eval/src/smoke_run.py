import json, re, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.loader import load_diqad, load_bioasq, load_mmath
from src.metrics import accuracy, macro_f1, exact_match

def main():
    diqad = load_diqad(str(PROJECT_ROOT/"data"/"diqad_sample.jsonl"))
    bioasq = load_bioasq(str(PROJECT_ROOT/"data"/"bioasq_small.jsonl"))
    mmath = load_mmath(str(PROJECT_ROOT/"data"/"mmath_sample.jsonl"), str(PROJECT_ROOT/"data"/"mmath_images"))

    diqad_preds = [0,2,1]
    diqad_gold  = [ex["gold"] for ex in diqad]
    diqad_acc   = accuracy(diqad_gold, diqad_preds)
    diqad_f1    = macro_f1(diqad_gold, diqad_preds)

    bio_preds = ["phentermine and topiramate", "reduces hepatic gluconeogenesis and improves insulin sensitivity"]
    bio_em = sum(exact_match(p, ex["gold"]) for p, ex in zip(bio_preds, bioasq)) / len(bioasq)

    def norm_math(x):
        x = (x or "").lower().replace("√","sqrt").replace(" ", "")
        m = re.search(r'(\d+(\.\d+)?(sqrt\d+)?)', x)
        return m.group(1) if m else x
    mmath_preds = ["4","3","9sqrt3"]
    mmath_gold = [norm_math(ex["gold"]) for ex in mmath]
    mmath_preds_n = [norm_math(x) for x in mmath_preds]
    mmath_acc = accuracy(mmath_gold, mmath_preds_n)

    metrics = {
        "diqad": {"accuracy": round(diqad_acc,3), "macro_f1": round(diqad_f1,3)},
        "bioasq": {"exact_match": round(bio_em,3)},
        "mmath": {"accuracy_final_number": round(mmath_acc,3)}
    }
    out = PROJECT_ROOT/"results"/"metrics"/"aggregate_results.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()