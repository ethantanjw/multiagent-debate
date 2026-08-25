"""
Base module with shared model querying and initialization functions.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from .commons import query_model, query_hf_model, load_model_tokenizer, get_vllm_client


def initialize_models(model_name, group_model=None, adv_model=None, gpus="0"):
    """
    Initialize models based on the provided names.

    Args:
        model_name: Model name for standard mode
        group_model: Model name for honest agents in adversarial mode
        adv_model: Model name for adversarial agents
        gpus: GPU devices to use

    Returns:
        dict with initialized models and clients
    """
    import os
    import sys
    # Add parent directory to path to import config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    models = {}

    # Check if using vLLM
    use_vllm = getattr(config, 'USE_VLLM', False)

    if use_vllm:
        print("Using vLLM server for inference")
        models['client'] = get_vllm_client()
        models['type'] = 'vllm'
        if model_name:
            models['model_name'] = model_name
        if group_model:
            models['group_model_name'] = group_model
        if adv_model:
            models['adv_model_name'] = adv_model
        return models

    # Legacy path: Load models locally
    # Standard mode
    if model_name:
        if "mistral" in model_name or "llama" in model_name or "Yi" in model_name or "Qwen" in model_name:
            model, tokenizer = load_model_tokenizer(model_name)
            models['model'] = model
            models['tokenizer'] = tokenizer
            models['type'] = 'hf'
        elif "gpt" in model_name:
            models['client'] = OpenAI()
            models['type'] = 'openai'
            models['model_name'] = model_name
        else:
            raise ValueError(f"Model {model_name} not supported")

    # Adversarial mode
    if group_model or adv_model:
        if "gpt" in (group_model or "") or "gpt" in (adv_model or ""):
            models['client'] = OpenAI()

        if group_model:
            if "mistral" in group_model or "llama" in group_model or "Yi" in group_model or "Qwen" in group_model:
                group_model_obj, group_tokenizer = load_model_tokenizer(group_model)
                models['group_model'] = group_model_obj
                models['group_tokenizer'] = group_tokenizer
            models['group_model_name'] = group_model

        if adv_model:
            if "mistral" in adv_model or "llama" in adv_model or "Yi" in adv_model or "Qwen" in adv_model:
                adv_model_obj, adv_tokenizer = load_model_tokenizer(adv_model)
                models['adv_model'] = adv_model_obj
                models['adv_tokenizer'] = adv_tokenizer
            models['adv_model_name'] = adv_model

    return models


def query_agent(agent_context, models, is_adversary=False):
    """
    Query a single agent with their context.

    Args:
        agent_context: The conversation context for this agent
        models: Dictionary of initialized models
        is_adversary: Whether this is an adversarial agent

    Returns:
        Completion string from the model
    """
    # Standard mode
    if 'type' in models and models['type'] == 'openai':
        return query_model(models['client'], agent_context, models['model_name'])
    elif 'type' in models and models['type'] == 'hf':
        return query_hf_model(models['model'], models['tokenizer'], agent_context)

    # Adversarial mode
    if is_adversary:
        model_name = models.get('adv_model_name')
        if "gpt" in model_name:
            return query_model(models['client'], agent_context, model_name)
        else:
            return query_hf_model(models['adv_model'], models['adv_tokenizer'], agent_context)
    else:
        model_name = models.get('group_model_name')
        if "gpt" in model_name:
            return query_model(models['client'], agent_context, model_name)
        else:
            return query_hf_model(models['group_model'], models['group_tokenizer'], agent_context)
