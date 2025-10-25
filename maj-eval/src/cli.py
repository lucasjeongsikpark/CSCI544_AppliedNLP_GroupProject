
import argparse
from . import generate, judge, score, report

def main():
    ap = argparse.ArgumentParser(description="MAJ-EVAL end-to-end pipeline")
    ap.add_argument("--config", default="config/run_config.yaml", help="Path to YAML config")
    ap.add_argument("cmd", choices=["generate","judge","score","report","all"])
    args = ap.parse_args()

    if args.cmd=="generate":
        generate.run(args.config)
    elif args.cmd=="judge":
        judge.run(args.config)
    elif args.cmd=="score":
        score.run(args.config)
    elif args.cmd=="report":
        report.run(args.config)
    elif args.cmd=="all":
        generate.run(args.config)
        judge.run(args.config)
        score.run(args.config)
        report.run(args.config)

if __name__ == "__main__":
    main()
