import argparse
import json
import os
from typing import Dict
from debate_runtime.orchestrator import DebateConfig, DebateOrchestrator
from debate_runtime.runner_debate import load_model, PROVIDER_CLASS  # reuse loader
from src.util import setup_default_logger

"""Run debates over a dataset file where each item supplies a topic and an initial context.

Config schema example:
{
  "dataset_path": "datasets/data/math_cleaned_250.json",
  "topic_field": "input",
  "context_field": "distill_llama_output",  # ANSWER_A in debate
  "secondary_context_field": "llama_output",  # ANSWER_B in debate (optional)
  "ground_truth_field": "output",  # for domain-specific evaluation
  "domain": "math",  # enables domain-specific eval prompt: 'math', 'medical', or 'openqa'
  "max_items": 3,
  "debate": {
    "max_rounds": 2,
    "roles_order": ["AFFIRMATIVE", "NEGATIVE"],
    "judge_role": "JUDGE",
    "temperature": 0.7,
    "max_tokens_generation": 400,
    "max_tokens_judge": 256
  },
  "generation_models": {
    "AFFIRMATIVE": {"provider": "ollama", "engine": "llama3.1"},
    "NEGATIVE": {"provider": "ollama", "engine": "llama3.1"}
  },
  "judge_model": {"provider": "ollama", "engine": "llama3.1"},
  "output_dir": "debate_outputs_dataset"
}
"""


from typing import Tuple

def build_models(gen_spec: Dict, judge_spec: Dict) -> Tuple[Dict, object]:
    gen_models = {}
    for role, model_spec in gen_spec.items():
        gen_models[role] = load_model(model_spec['provider'], model_spec['engine'], model_spec.get('key_path'))
    judge_model = load_model(judge_spec['provider'], judge_spec['engine'], judge_spec.get('key_path'))
    return gen_models, judge_model


def main():
    parser = argparse.ArgumentParser(description='Run debates across dataset entries.')
    parser.add_argument('--config', required=True, help='Path to dataset debate config JSON')
    args = parser.parse_args()
    config = json.load(open(args.config))

    output_dir = config.get('output_dir', 'debate_outputs_dataset')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_default_logger(output_dir)

    dataset_path = config['dataset_path']
    topic_field = config.get('topic_field', 'input')
    context_field = config.get('context_field')  # may be None
    secondary_context_field = config.get('secondary_context_field')  # optional second answer field
    ground_truth_field = config.get('ground_truth_field', 'output')  # for domain eval
    domain = config.get('domain')  # 'math', 'medical', 'openqa', or None
    max_items = config.get('max_items')
    start_index = config.get('start_index', 0)  # optional starting index

    with open(dataset_path) as f:
        data = json.load(f)

    if max_items is not None:
        data = data[start_index:start_index + max_items]
    else:
        data = data[start_index:]

    gen_models, judge_model = build_models(config['generation_models'], config['judge_model'])

    # Define output file path
    output_file = os.path.join(output_dir, 'debates_consolidated.json')
    
    # Load existing results if file exists (for resuming)
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_results = json.load(f)
        logger.info(f"Loaded {len(all_results)} existing results from {output_file}")
    else:
        all_results = []
    
    # Iterate entries
    for idx, item in enumerate(data, start=start_index):
        topic = item.get(topic_field, f"Item {idx}")
        initial_context = item.get(context_field, '') if context_field else ''
        secondary_context = item.get(secondary_context_field, '') if secondary_context_field else ''
        ground_truth = item.get(ground_truth_field, '')

        debate_conf = DebateConfig(
            topic=topic[:512],  # truncate overly long topics
            max_rounds=config['debate'].get('max_rounds', 2),
            roles_order=config['debate'].get('roles_order', ['AFFIRMATIVE','NEGATIVE']),
            judge_role=config['debate'].get('judge_role', 'JUDGE'),
            temperature=config['debate'].get('temperature', 0.7),
            max_tokens_generation=config['debate'].get('max_tokens_generation', 400),
            max_tokens_judge=config['debate'].get('max_tokens_judge', 256),
            initial_context=initial_context[:3000],  # safety truncation
            secondary_context=secondary_context[:3000],
            domain=domain,
            ground_truth=ground_truth[:2000] if ground_truth else '',
            system_prompt=item.get('system_prompt', ''),  # for openqa
            document=item.get('document', '')  # for medical
        )

        orchestrator = DebateOrchestrator(debate_conf, gen_models, judge_model, logger=logger)
        state = orchestrator.run()
        
        # Parse JSON and add to results array
        result_dict = json.loads(state.to_json())
        all_results.append(result_dict)
        
        # Write to file immediately after each debate (incremental save)
        with open(output_file, 'w') as f_out:
            json.dump(all_results, f_out, indent=2)
        
        logger.info(f'Completed debate {idx} topic snippet=[{topic[:80]}] - saved to {output_file}')

    logger.info(f'All debates completed. Total results: {len(all_results)} in {output_file}')

    logger.info('All debates completed.')

if __name__ == '__main__':
    main()
