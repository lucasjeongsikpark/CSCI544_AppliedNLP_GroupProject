import argparse
import os
import time
from typing import Dict
from model import LanguageModel, OpenAIModel, HuggingFaceModel, AnthropicModel, OpenAIReasoningModel, OllamaModel
from util import setup_default_logger, create_out_dir
import json
import pandas as pd


def parse_response(response: str) -> int:
    if '</scratchpad>' in response:
        response = response.split('</scratchpad>')[1].strip()

    response_start = response.find('<score>') + len('<score>')
    response_end = response.find('</score>')
    if response_start != -1 and response_end != -1:
        parsed = response[response_start:response_end]
    else:
        parsed = response.replace('<score>', '').replace('</score>', '')
        parsed = parsed.replace('score', '').replace('Score', '')
        parsed = parsed.replace(':', '')
    parsed = parsed.strip()
    if parsed in ['1', '2', '3', '4', '5']:
        return int(parsed)

    logger.error(f'Parsing failure response=[{response}], parsed=[{parsed}]')
    return -1


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


def dataset_experiment():
    dataset_type = config.get('dataset_type', 'unknown')
    logger.info(f'Running experiment for dataset type: {dataset_type}')
    

    # Read dataset based on type
    data = read_dataset(config['data_path'], dataset_type)
    logger.info(f'Loaded {len(data)} rows from {config["data_path"]}')

    experiments_config = config['experiments']
    max_tokens = int(config['max_tokens'])
    temperature = float(config['temperature'])

    # Create output directory
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

            # Generate prompts for both llama_output and distill_llama_output
            prompts_dict = generate_prompts_for_dataset(exp_data, prompt_template, dataset_type)
            prompts = {k: v['prompt'] for k, v in prompts_dict.items()}

            batch_idx = model.request_batch_completions(prompts, max_tokens, temperature, 0, config['output_path'])

            logger.info(f'Waiting for batch {batch_idx} to complete...')
            responses = model.get_batch_completions(batch_idx)
            while not responses:
                time.sleep(60)  # Wait for 1 minute
                responses = model.get_batch_completions(batch_idx)
                logger.info(f'Waiting for batch {batch_idx} to complete...')

            experiment_results[model_name] = {}
            experiment_name = experiment["name"]

            # Collect results for both answer types
            results = []
            for k, v in prompts_dict.items():
                row_id = v['id']
                answer_type = v['answer_type']
                response = responses[k]['completion'] if k in responses else ''
                score = parse_response(response)
                reasoning = get_reasoning(response)
                # Find the original row for context
                orig_row = exp_data[exp_data['id'] == row_id].iloc[0]
                results.append({
                    'id': row_id,
                    'answer_type': answer_type,
                    'input': orig_row.get('input', ''),
                    'output': orig_row.get('output', ''),
                    'model': model_name,
                    'experiment': experiment_name,
                    'score': score,
                    'response': response,
                    'reasoning': reasoning
                })
                # Logging
                log_string = '\n==================================\n'
                if dataset_type == 'math':
                    log_string += f'----------PROBLEM----------\n{orig_row["input"]}'
                    log_string += f'\n----------EXPECTED----------\n{orig_row["output"]}'
                elif dataset_type == 'openqa':
                    log_string += f'----------QUESTION----------\n{orig_row["input"]}'
                    log_string += f'\n----------EXPECTED----------\n{orig_row["output"]}'
                elif dataset_type == 'medical':
                    log_string += f'----------PATIENT QUESTION----------\n{orig_row["input"]}'
                    log_string += f'\n----------REFERENCE RESPONSE----------\n{orig_row["output"]}'
                log_string += f'\n----------ANSWER TYPE----------\n{answer_type}'
                log_string += '\n---------------------------------\n'
                log_string += f'MODEL-[{model_name}] SCORE: {score}'
                if reasoning:
                    log_string += f'\n----------REASONING----------\n{reasoning}'
                logger.info(log_string)

            # Save results to DataFrame and CSV
            results_df = pd.DataFrame(results)
            output_path = os.path.join(config['output_path'], f'{experiment["name"]}_{model_name}_results.csv')
            results_df.to_csv(output_path, index=False)
            logger.info(f'Saved predictions to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    logger = setup_default_logger(config['output_path'])
    logger.info(f'Starting experiment with config: {args.config}')

    dataset_experiment()
