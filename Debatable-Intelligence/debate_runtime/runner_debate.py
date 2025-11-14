import argparse
import json
import os
from debate_runtime.orchestrator import DebateConfig, DebateOrchestrator
from src.util import setup_default_logger
from src.model import OpenAIModel, OpenAIReasoningModel, HuggingFaceModel, AnthropicModel, OllamaModel

# Minimal mapping helper; extend for more providers
PROVIDER_CLASS = {
    'openai': OpenAIModel,
    'openai_reasoning': OpenAIReasoningModel,
    'huggingface': HuggingFaceModel,
    'anthropic': AnthropicModel,
    'ollama': OllamaModel,
}

def load_model(provider: str, engine: str, key_path: str | None):
    if provider == 'ollama':
        return OllamaModel(engine)
    key = ''
    if key_path:
        with open(key_path, 'r') as f:
            key = f.read().strip()
    if provider == 'openai_reasoning':
        return OpenAIReasoningModel(key, engine)
    return PROVIDER_CLASS[provider](key, engine)


def main():
    parser = argparse.ArgumentParser(description='Run interactive debate using Debatable-Intelligence judging.')
    parser.add_argument('--config', required=True, help='Path to debate_config.json')
    args = parser.parse_args()

    config = json.load(open(args.config))
    output_dir = config.get('output_dir', 'debate_outputs')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_default_logger(output_dir)

    debate_conf = DebateConfig(
        topic=config['topic'],
        max_rounds=config.get('max_rounds', 3),
        roles_order=config.get('roles_order', ['AFFIRMATIVE', 'NEGATIVE']),
        judge_role=config.get('judge_role', 'JUDGE'),
        temperature=config.get('temperature', 0.7),
        max_tokens_generation=config.get('max_tokens_generation', 512),
        max_tokens_judge=config.get('max_tokens_judge', 256),
    )

    # Load generation models per role
    gen_models = {}
    for role, model_spec in config['generation_models'].items():
        gen_models[role] = load_model(model_spec['provider'], model_spec['engine'], model_spec.get('key_path'))

    judge_spec = config['judge_model']
    judge_model = load_model(judge_spec['provider'], judge_spec['engine'], judge_spec.get('key_path'))

    orchestrator = DebateOrchestrator(debate_conf, gen_models, judge_model, logger=logger)
    state = orchestrator.run()

    out_file = os.path.join(output_dir, 'debate_state.json')
    with open(out_file, 'w') as f:
        f.write(state.to_json())
    logger.info(f'Saved debate state to {out_file}')

if __name__ == '__main__':
    main()
