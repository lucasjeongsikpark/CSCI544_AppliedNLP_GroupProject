import argparse
import os
import time
from typing import Dict, Tuple, Any, List
from model import LanguageModel, OpenAIModel, HuggingFaceModel, AnthropicModel, OpenAIReasoningModel, OllamaModel
from util import setup_default_logger, create_out_dir
import json
import pandas as pd


def parse_response(response: str) -> Tuple[int, Dict[str, Any]]:
    """Parse model response extracting overall score and per-metric scores.

    Returns (overall_score, metrics_dict).
    Backward compatible: if only legacy <score> tag present metrics_dict will be empty.
    New expected format:
    <metrics>\nmetric: val\n...</metrics> and <overall_score>val</overall_score>
    """
    metrics: Dict[str, Any] = {}
    text = response
    # Extract metrics block
    if '<metrics>' in text and '</metrics>' in text:
        mstart = text.find('<metrics>') + len('<metrics>')
        mend = text.find('</metrics>')
        block = text[mstart:mend].strip()
        for line in block.splitlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            k = k.strip().lower()
            v = v.strip().replace('[','').replace(']','')
            # keep only integer 1-5
            if v.isdigit():
                iv = int(v)
                if iv in (1,2,3,4,5):
                    metrics[k] = iv
    # Extract overall score
    overall = -1
    if '<overall_score>' in text and '</overall_score>' in text:
        s = text.find('<overall_score>') + len('<overall_score>')
        e = text.find('</overall_score>')
        raw = text[s:e].strip().replace('[','').replace(']','')
        if raw.isdigit():
            iv = int(raw)
            if iv in (1,2,3,4,5):
                overall = iv
    else:
        # legacy single-score
        s = text.find('<score>') + len('<score>')
        e = text.find('</score>')
        if s != -1 and e != -1:
            raw = text[s:e].strip().replace('[','').replace(']','')
            if raw.isdigit():
                iv = int(raw)
                if iv in (1,2,3,4,5):
                    overall = iv
        else:
            cleaned = text.replace('<score>', '').replace('</score>', '').replace(':','').strip()
            if cleaned.isdigit():
                iv = int(cleaned)
                if iv in (1,2,3,4,5):
                    overall = iv
    if overall == -1:
        logger.error(f'Parsing failure overall score response=[{response[:200]}...]')
    return overall, metrics


def get_reasoning(response: str) -> str:
    reasoning_start = response.find('<scratchpad>') + len('<scratchpad>')
    reasoning_end = response.find('</scratchpad>')
    if reasoning_start != -1 and reasoning_end != -1:
        return response[reasoning_start:reasoning_end].strip()
    return ''


def read_dataset(data_path: str, dataset_type: str) -> pd.DataFrame:
    """Read dataset based on type and add necessary ID column"""
    df = pd.read_csv(data_path)
    
    # Add ID column if not present
    if 'id' not in df.columns:
        df['id'] = df.index.astype(str)
    
    return df


def generate_prompts_for_dataset(df: pd.DataFrame, prompt_template: str, dataset_type: str) -> Dict[str, Dict[str, str]]:
    """Generate prompts for both llama_output and distill_llama_output for each row."""
    prompts = {}
    for idx, row in df.iterrows():
        for answer_type in ['llama_output', 'distill_llama_output']:
            prompt = prompt_template
            if dataset_type == 'math':
                prompt = prompt.replace('{INPUT}', str(row['input']))
                prompt = prompt.replace('{OUTPUT}', str(row['output']))
                prompt = prompt.replace('{RESPONSE}', str(row.get(answer_type, '[TO BE GENERATED]')))
            elif dataset_type == 'openqa':
                system_prompt = str(row.get('system_prompt', ''))
                prompt = prompt.replace('{SYSTEM_PROMPT}', system_prompt)
                prompt = prompt.replace('{INPUT}', str(row['input']))
                prompt = prompt.replace('{OUTPUT}', str(row['output']))
                prompt = prompt.replace('{RESPONSE}', str(row.get(answer_type, '[TO BE GENERATED]')))
            elif dataset_type == 'medical':
                document = str(row.get('document', ''))
                prompt = prompt.replace('{DOCUMENT}', document)
                prompt = prompt.replace('{INPUT}', str(row['input']))
                prompt = prompt.replace('{OUTPUT}', str(row['output']))
                prompt = prompt.replace('{RESPONSE}', str(row.get(answer_type, '[TO BE GENERATED]')))
            prompts[f"{row['id']}__{answer_type}"] = {
                'prompt': prompt,
                'id': row['id'],
                'answer_type': answer_type
            }
    return prompts


def setup_models() -> Dict[str, LanguageModel]:
    models = {}
    participating_models = config['participating_models']
    available_models = config['models']
    
    if 'openai' in participating_models:
        with open(config['openai_key_path'], 'r') as f:
            api_key = f.read().strip()
        assert api_key, 'OpenAI API key is empty'
        for engine in available_models['openai']:
            if engine.startswith('gpt-4'):
                logger.info(f'Setting up OpenAI model with engine: {engine}')
                models[f'openai_{engine}'] = OpenAIModel(api_key, engine)
            else:
                logger.info(f'Setting up OpenAI reasoning model with engine: {engine}')
                models[f'openai_{engine}'] = OpenAIReasoningModel(api_key, engine)
    
    if 'huggingface' in participating_models:
        with open(config['huggingface_key_path'], 'r') as f:
            api_key = f.read().strip()
        assert api_key, 'HuggingFace API key is empty'
        for model_name in available_models['huggingface']:
            logger.info(f'Setting up HuggingFace model with model name: {model_name}')
            models[f'huggingface_{model_name}'] = HuggingFaceModel(api_key, model_name)
    
    if 'anthropic' in participating_models:
        with open(config['anthropic_key_path'], 'r') as f:
            api_key = f.read().strip()
        assert api_key, 'Anthropic API key is empty'
        for model_name in available_models['anthropic']:
            logger.info(f'Setting up Anthropic model with model name: {model_name}')
            models[f'anthropic_{model_name}'] = AnthropicModel(api_key, model_name)

    if 'ollama' in participating_models:
        base_url = config.get('ollama_base_url', 'http://localhost:11434')
        for model_name in available_models.get('ollama', []):
            logger.info(f'Setting up Ollama model with name: {model_name} (base_url={base_url})')
            models[f'ollama_{model_name}'] = OllamaModel(model_name, base_url)

    return models


REQUIRED_METRICS = {
    'math': ['correctness', 'reasoning', 'completeness', 'accuracy'],
    'openqa': ['relevance', 'completeness', 'accuracy', 'clarity', 'helpfulness'],  # example; adjust if needed
    'medical': ['relevance', 'completeness', 'accuracy', 'clarity', 'helpfulness']  # example; adjust if needed
}

def has_required_metrics(dataset_type: str, metrics: Dict[str, Any]) -> bool:
    req = REQUIRED_METRICS.get(dataset_type, [])
    return all(m in metrics and isinstance(metrics[m], int) for m in req)

def dataset_experiment():
    dataset_type = config.get('dataset_type', 'unknown')
    logger.info(f'Running experiment for dataset type: {dataset_type}')
    

    # Read dataset based on type
    data = read_dataset(config['data_path'], dataset_type)
    logger.info(f'Loaded {len(data)} rows from {config["data_path"]}')

    experiments_config = config['experiments']
    max_tokens = int(config['max_tokens'])
    temperature = float(config['temperature'])

    # Create output directory (domain folder)
    create_out_dir(config['output_path'])

    models = setup_models()


    for experiment in experiments_config:
        if experiment['run'] is False:
            continue

        logger.info(f'Running experiment: {experiment["name"]}')
        with open(experiment['prompt_path'], 'r') as f:
            prompt_template = f.read()

        # Respect max_examples if set in experiment config
        exp_data = data
        max_examples = experiment.get('max_examples', None)
        if max_examples is not None:
            exp_data = exp_data.iloc[:int(max_examples)]
            logger.info(f'Limiting to first {max_examples} examples for this experiment.')

        experiment_results = {}
        for model_name, model in models.items():
            logger.info(f'Running experiment for model: {model_name}')

            # Create subdirectory for this model under the output/domain folder
            model_output_dir = os.path.join(config['output_path'], model_name)
            create_out_dir(model_output_dir)

            # Generate prompts for both llama_output and distill_llama_output
            prompts_dict = generate_prompts_for_dataset(exp_data, prompt_template, dataset_type)
            prompts = {k: v['prompt'] for k, v in prompts_dict.items()}

            batch_idx = model.request_batch_completions(prompts, max_tokens, temperature, 0, model_output_dir)

            logger.info(f'Waiting for batch {batch_idx} to complete...')
            responses = model.get_batch_completions(batch_idx)
            while not responses:
                time.sleep(60)  # Wait for 1 minute
                responses = model.get_batch_completions(batch_idx)
                logger.info(f'Waiting for batch {batch_idx} to complete...')

            experiment_results[model_name] = {}
            experiment_name = experiment["name"]

            # Aggregate per original row id combining both answer types into single JSON entry
            grouped_ids = {}
            for key, meta in prompts_dict.items():
                grouped_ids.setdefault(meta['id'], []).append(key)

            json_entries: List[Dict[str, Any]] = []
            csv_rows: List[Dict[str, Any]] = []
            max_attempts = int(experiment.get('max_attempts', config.get('max_attempts', 3)))

            for row_id, keys in grouped_ids.items():
                start_time = time.time()
                # Find original row
                try:
                    orig_row = exp_data[exp_data['id'] == row_id].iloc[0]
                except Exception:
                    orig_row = {}

                entry = {
                    'id': row_id,
                    'input': orig_row.get('input', '') if isinstance(orig_row, dict) else orig_row.get('input', ''),
                    'output': orig_row.get('output', '') if isinstance(orig_row, dict) else orig_row.get('output', ''),
                    'model': model_name,
                    'experiment': experiment_name,
                    'score1': {},
                    'score2': {},
                    'chat_log': {},
                    'attempts': 0,
                }
                total_attempts = 0

                for answer_key in keys:
                    answer_type = prompts_dict[answer_key]['answer_type']
                    prompt_text = prompts_dict[answer_key]['prompt']
                    attempt_count = 0
                    response_text = ''
                    metrics_ok = False
                    response_obj = responses.get(answer_key, {})
                    # Attempt loop: re-query if structure invalid
                    while attempt_count < max_attempts:
                        attempt_count += 1
                        if not response_obj:
                            logger.error(f'Missing response object for key={answer_key} row_id={row_id} attempt={attempt_count}')
                            response_text = ''
                        else:
                            response_text = response_obj.get('completion', '') or ''
                        overall_score, metric_scores = parse_response(response_text)
                        reasoning = get_reasoning(response_text)
                        metrics_ok = (overall_score != -1) and has_required_metrics(dataset_type, metric_scores)
                        if metrics_ok:
                            break
                        # If not ok and attempts remain, re-query
                        if attempt_count < max_attempts:
                            logger.info(f'Re-attempting key={answer_key} (attempt {attempt_count+1}) due to missing/invalid structure metrics_ok={metrics_ok}')
                            try:
                                single_prompt = {answer_key: prompt_text}
                                single_response_path = model.request_batch_completions(single_prompt, max_tokens, temperature, 9999, model_output_dir)
                                with open(single_response_path, 'r') as f:
                                    new_completions = json.load(f)
                                if answer_key in new_completions:
                                    response_obj = new_completions[answer_key]
                            except Exception as e:
                                logger.error(f'Retry failed for key={answer_key} error={e}')

                    total_attempts += attempt_count
                    # Record scores
                    scores_dict = {}
                    for mk in REQUIRED_METRICS.get(dataset_type, metric_scores.keys()):
                        if mk in metric_scores:
                            scores_dict[mk.capitalize()] = metric_scores[mk]
                    # Fallback overall if metrics missing
                    if not scores_dict and overall_score != -1:
                        scores_dict['Overall'] = overall_score
                    # Store
                    if answer_type == 'llama_output':
                        entry['score1'] = scores_dict
                    else:
                        entry['score2'] = scores_dict
                    entry['chat_log'][answer_type] = {
                        'prompt': prompt_text,
                        'response': response_text,
                        'reasoning': get_reasoning(response_text)
                    }

                elapsed_time = time.time() - start_time
                entry['attempts'] = total_attempts
                entry['elapsed_time'] = elapsed_time
                json_entries.append(entry)

                # CSV row (flatten)
                flat_row = {
                    'id': entry['id'],
                    'model': entry['model'],
                    'experiment': entry['experiment'],
                    'attempts': entry['attempts'],
                    'elapsed_time': entry['elapsed_time']
                }
                for side, scores in [('score1', entry['score1']), ('score2', entry['score2'])]:
                    for mk, mv in scores.items():
                        flat_row[f'{side}_{mk.lower()}'] = mv
                csv_rows.append(flat_row)

            # Persist outputs
            results_df = pd.DataFrame(csv_rows)
            csv_output_path = os.path.join(model_output_dir, f'{experiment["name"]}_{model_name}_results.csv')
            results_df.to_csv(csv_output_path, index=False)
            logger.info(f'Saved aggregated CSV predictions to {csv_output_path}')
            json_output_path = os.path.join(model_output_dir, f'{experiment["name"]}_{model_name}_results.json')
            with open(json_output_path, 'w') as jf:
                json.dump(json_entries, jf, indent=2, ensure_ascii=False)
            logger.info(f'Saved JSON predictions to {json_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    logger = setup_default_logger(config['output_path'])
    logger.info(f'Starting experiment with config: {args.config}')

    dataset_experiment()
