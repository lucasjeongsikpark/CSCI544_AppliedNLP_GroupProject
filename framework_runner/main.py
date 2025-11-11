import argparse
import re
from typing import List

from .debate_impl import debate_framework
from .debint_impl import debint_framework
from .runner import FrameworkRunner


FRAMEWORKS = {
    debate_framework.name.lower(): debate_framework,
    debint_framework.name.lower(): debint_framework,
}

DEFAULT_ASPECTS = {
    'math': ['correctness', 'reasoning', 'completeness', 'accuracy'],
    'openqa': ['relevance', 'completeness', 'accuracy', 'clarity', 'helpfulness'],
    'medical': ['medical_accuracy', 'appropriateness', 'safety', 'clarity', 'professionalism'],
}


def parse_aspects_file(path: str) -> List[str]:
    aspects: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    capture = False
    for line in lines:
        if 'Metrics:' in line:
            capture = True
            continue
        if capture and line.strip().startswith('- '):
            # pattern: - correctness: Description...
            m = re.match(r'-\s*([A-Za-z_ ]+):', line.strip())
            if m:
                aspects.append(m.group(1).strip().lower())
        # Stop if we reach Output format section
        if 'Output format' in line:
            break
    return aspects


def main():
    parser = argparse.ArgumentParser(description="Run a scoring framework over a dataset and produce JSONL output.")
    parser.add_argument('--framework', type=str, required=True, help='Framework name (debate or debint)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset (.json or .csv)')
    parser.add_argument('--output_file', type=str, required=True, help='NDJSON/JSONL output path')
    parser.add_argument('--aspects', type=str, default='', help='Comma separated aspects override')
    parser.add_argument('--aspects_file', type=str, default='', help='Prompt file to parse aspects from')
    parser.add_argument('--dataset_type', type=str, default='', help='Hint for default aspects (math|openqa|medical) if none provided')
    parser.add_argument('--start_from', type=int, default=0, help='Row index to start from (resume)')
    parser.add_argument('--no_auto_resume', action='store_true', help='Disable auto resume based on existing output')
    parser.add_argument('--limit', type=int, default=None, help='Process only this many rows (for debugging)')
    args = parser.parse_args()

    fw_key = args.framework.lower()
    if fw_key not in FRAMEWORKS:
        raise ValueError(f'Unknown framework {args.framework}. Available: {list(FRAMEWORKS.keys())}')
    framework = FRAMEWORKS[fw_key]

    # Determine aspects priority: explicit list > aspects_file > dataset_type defaults
    if args.aspects:
        aspects = [a.strip().lower() for a in args.aspects.split(',') if a.strip()]
    elif args.aspects_file:
        aspects = parse_aspects_file(args.aspects_file)
    elif args.dataset_type and args.dataset_type.lower() in DEFAULT_ASPECTS:
        aspects = DEFAULT_ASPECTS[args.dataset_type.lower()]
    else:
        aspects = []
    if not aspects:
        raise ValueError('No aspects determined. Provide --aspects, --aspects_file, or --dataset_type.')

    runner = FrameworkRunner(
        framework=framework,
        dataset_path=args.dataset_path,
        output_file=args.output_file,
        aspects=aspects,
        start_from=args.start_from,
        auto_resume=not args.no_auto_resume,
        limit=args.limit,
    )
    runner.evaluate_dataset()


if __name__ == '__main__':
    main()
