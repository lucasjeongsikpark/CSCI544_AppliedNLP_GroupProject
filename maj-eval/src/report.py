import os, json, yaml, datetime

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    metrics_dir = cfg["paths"]["metrics_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    _ensure_dir(reports_dir)

    agg_path = os.path.join(metrics_dir, "aggregate_results.json")
    if not os.path.exists(agg_path):
        print("No metrics found. Run 'score' first.")
        return
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)

    md = ["# MAJ-EVAL Replication Report",
          f"_Generated: {datetime.datetime.utcnow().isoformat()}Z_",
          "",
          "## Summary Metrics"]
    for ds, models in agg.items():
        md.append(f"### {ds}")
        for m, vals in models.items():
            s = ", ".join([f"{k}: {v:.3f}" for k,v in vals.items()])
            md.append(f"- **{m}** — {s}")
        md.append("")
    out_md = os.path.join(reports_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as w:
        w.write("\n".join(md))
    print(f"Wrote report -> {out_md}")
