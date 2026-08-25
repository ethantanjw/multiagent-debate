import time
import asyncio
from .math_parsing import parse_math
from .prompt import agent_prompt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI



def query_model(client, agent_context, model_name="gpt-4o-mini"):
    """
    Query OpenAI using the new Responses API for GPT-5 models (synchronous).
    Supports both chat format (messages) and single input format.
    """
    try:
        # Convert messages format to single input if needed
        if isinstance(agent_context, list):
            # Combine all messages into a single input string
            input_text = "\n\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in agent_context
            ])
        else:
            input_text = agent_context

        response = client.responses.create(
                model=model_name,
                input=input_text)

        content = response.output_text
        return content

    except Exception as e:
        print(f"API Error: {e}")
        print("Retrying in 20 seconds...")
        time.sleep(20)
        return query_model(client, agent_context, model_name)


async def query_model_async(client, agent_context, model_name="gpt-4o-mini"):
    """
    Async version of query_model for parallel execution across multiple questions.
    Supports both OpenAI Responses API and vLLM Chat Completions API.
    """
    try:
        # Detect if using vLLM (based on base_url containing localhost or port 8000)
        is_vllm = hasattr(client, 'base_url') and client.base_url and (
            'localhost' in str(client.base_url) or ':8000' in str(client.base_url)
        )

        if is_vllm:
            # vLLM uses Chat Completions API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=agent_context if isinstance(agent_context, list) else [{"role": "user", "content": agent_context}],
                    max_tokens=4096,  # Large context window allows generous token limits
                    temperature=0.7,
                )
            )
            return response.choices[0].message.content
        else:
            # OpenAI Responses API
            # Convert messages format to single input if needed
            if isinstance(agent_context, list):
                input_text = "\n\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in agent_context
                ])
            else:
                input_text = agent_context

            # Use asyncio to run the blocking API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.responses.create(model=model_name, input=input_text)
            )

            content = response.output_text
            return content

    except Exception as e:
        print(f"API Error: {e}")
        print("Retrying in 5 seconds...")
        await asyncio.sleep(5)
        return await query_model_async(client, agent_context, model_name)


def query_hf_model(model, tokenizer, agent_context):
    input_ids = tokenizer.apply_chat_template(
        agent_context,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)


    terminators = [tokenizer.eos_token_id]
    if "llama" in model.name_or_path.lower() or "gpt" in model.name_or_path.lower():
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        # temperature=0.6,
        # top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def query_vllm(vllm_client, agent_context, model_name):
    """
    Query vLLM server using OpenAI-compatible API.

    Args:
        vllm_client: OpenAI client configured for vLLM server
        agent_context: List of messages in chat format
        model_name: Model name (must match vLLM server model)

    Returns:
        Generated text response
    """
    try:
        response = vllm_client.chat.completions.create(
            model=model_name,
            messages=agent_context,
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"vLLM API Error: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        return query_vllm(vllm_client, agent_context, model_name)


async def query_vllm_async(vllm_client, agent_context, model_name):
    """
    Async version of query_vllm for parallel execution.
    """
    try:
        response = await vllm_client.chat.completions.create(
            model=model_name,
            messages=agent_context,
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"vLLM API Error: {e}")
        print("Retrying in 5 seconds...")
        await asyncio.sleep(5)
        return await query_vllm_async(vllm_client, agent_context, model_name)


def get_vllm_client():
    """
    Create OpenAI client configured for vLLM server.

    Returns:
        OpenAI client pointed at vLLM server
    """
    import sys
    import os
    # Add parent directory to path to import config
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

    # Compute API base if not explicitly set
    if config.VLLM_API_BASE is None:
        api_base = f"http://{config.VLLM_HOST}:{config.VLLM_PORT}/v1"
    else:
        api_base = config.VLLM_API_BASE

    return OpenAI(
        api_key=config.VLLM_API_KEY,
        base_url=api_base
    )


def load_model_tokenizer(model_name, device_map="auto"):
    """
    Load model and tokenizer from HuggingFace.

    Uses cache_dir from environment variable MODEL_CACHE_DIR if set,
    otherwise uses default HuggingFace cache.
    """
    import os

    # Get cache directory from environment (for cluster scratch storage)
    cache_dir = os.environ.get('MODEL_CACHE_DIR', None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        cache_dir=cache_dir
    )
    return model, tokenizer


def parse_question_answer(dataset_name, sample):
    
    if dataset_name == "mmlu":
        question_raw = sample[0]
        a = sample[1]
        b = sample[2]
        c = sample[3]
        d = sample[4]
        answer = sample[5]
        if type(sample) == list:
            raw_task = tuple(sample)
        else:
            raw_task = tuple(sample.values())
        question = agent_prompt[dataset_name]['question'].format(question_raw, a, b, c, d)
        return question, answer, raw_task
    
    elif dataset_name == "math":
        question_raw = sample["problem"]
        answer = parse_math(sample["solution"])
        question = agent_prompt[dataset_name]['question'].format(question_raw)
        raw_task = sample
        return question, answer, raw_task
    
    elif dataset_name == "chess":
        question_raw = sample["input"]
        last_move = sample['input'].split(' ')[-1]
        question = agent_prompt[dataset_name]['question'].format(question_raw, last_move)
        answer = sample["target"]
        raw_task = sample
        return question, answer, raw_task
    
    elif dataset_name == "mquake":
        question_raw = sample['questions'][0]
        answer = [sample['answer']] + sample['answer_alias']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw,question_raw)
        return question, answer, raw_task
    
    elif dataset_name == "musique":
        question_raw = sample['question']
        answer = [sample['answer']] + sample['answer_aliases']
        raw_task = sample
        question = agent_prompt[dataset_name]['question'].format(question_raw, question_raw)
        return question, answer, raw_task

    elif dataset_name == "truthfulqa":
        question_raw = sample['question']
        answers_raw = sample['mc1_targets']
        answers = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw)]
        answer = [(chr(97 + i), answer) for i, answer in enumerate(answers_raw) if answers_raw[answer] == 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in answers])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task
    
    elif dataset_name == "medmcqa":
        question_raw = sample['question']
        answers_letters = ['a', 'b', 'c', 'd']
        answers = [sample['opa'], sample['opb'], sample['opc'], sample['opd']]
        answer = answers_letters[sample['cop'] - 1]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in zip(answers_letters, answers)])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task
    
    elif dataset_name == 'scalr':

        question_raw = sample['question']
        answers_letters = ['a', 'b', 'c', 'd', 'e']
        answers = [sample['choice_0'], sample['choice_1'], sample['choice_2'], sample['choice_3'], sample['choice_4']]
        answer = answers_letters[sample['answer'] ]
        raw_task = sample
        answers_txt = ', '.join([f"({letter.upper()}) {answer}" for letter, answer in zip(answers_letters, answers)])
        question = agent_prompt[dataset_name]['question'].format(question_raw, answers_txt)
        return question, answer, raw_task
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    


def query_model_extra(client, agent_context, model_name="gpt-3.5-turbo-0125", logprobs=False, top_logprobs=None, max_tokens=None, n_repetitions=1):
    
    top_logprobs = None if not logprobs else top_logprobs

    try:
        completion = client.chat.completions.create(
                model=model_name,
                messages=agent_context,
                n=n_repetitions,
                logprobs=logprobs, 
                top_logprobs=top_logprobs,
                max_tokens=max_tokens 
                )
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return query_model_extra(agent_context)
    
    return completion