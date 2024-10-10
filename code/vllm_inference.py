import warnings
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from typing import Optional, Callable, List, Tuple, Union
import copy
import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from packaging import version
import jsonlines
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import argparse
import json
import re
import sys
import os
from tqdm import tqdm
import numpy as np
from transformers import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams

MAX_INT = sys.maxsize


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--size", type=str, default="1.3B")
    parser.add_argument("--data_file", type=str, default="arena")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=4
    )  # tensor_parallel_size
    return parser.parse_args()


if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    if args.model == "gpt":
        model_path = f"../../cerebras/Cerebras-GPT-{args.size}" 
    elif args.model == "pythia":
        model_path = f"../../EleutherAI/pythia-{args.size}"  
    elif args.model == "Llama-2":  
        model_path = f"../../meta-llama/Llama-2-{args.size}-chat-hf"  
    elif args.model == "Llama-3":  
        model_path = f"../../meta-llama/Meta-Llama-3-{args.size}-Instruct"  
    elif args.model == "deepseek":
        if args.size == "7b":
            model_path = f"../../deepseek-ai/deepseek-coder-{args.size}-instruct-v1.5"
        else:
            model_path = f"../../deepseek-ai/deepseek-coder-{args.size}-instruct"
    elif args.model == "qwen2":
        model_path = f"../../Qwen/Qwen2-{args.size}-Instruct"
    elif args.model == "qwen1.5":
        model_path = f"../../Qwen/Qwen1.5-{args.size}-Chat"
    elif args.model == "mistral":
        model_path = f"../../mistralai/Mistral-{args.size}-Instruct-v0.3"
    elif args.model == "gemma":
        model_path = f"../../google/gemma-{args.size}"
    elif args.model == "WizardLMTeam":
        if args.size == "13B":
            model_path = f"../../WizardLMTeam/WizardLM-{args.size}-V1.2"
        else:
            model_path = f"../../WizardLMTeam/WizardLM-{args.size}-V1.0"
    elif args.model == "vicuna":
        if args.size == "33B":
            model_path = f"../../lmsys/vicuna-{args.size}-v1.3"
        else:
            model_path = f"../../lmsys/vicuna-{args.size}-v1.5"
    else:
        pass


    model = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    questions = []
    results = []

    if "arena" in args.data_file:
        data_file = "data/arena-hard-data.json"
    elif "alpaca" in args.data_file:
        data_file = "data/test.json"
    elif "openocra" in args.data_file:
        data_file = "data/Open-Orca-data.json"
    with open(data_file, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            questions.append(item["instruction"])

    print("**********length***********:", len(questions))
    batch_questions = batch_data(questions, batch_size=args.batch_size)

    res_completions = []
    for idx, batch in enumerate(batch_questions):


        inputs = ""
        token_id = []
        for prompt in tqdm(batch, desc="Generating text", leave=False):
            messages = [
                {"role": "user", "content": prompt}
            ]

            input_ids = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]
            token_id.append(input_ids.tolist())

        sampling_params = SamplingParams(temperature=0.7, max_tokens=1000)
        completions = model.generate(
            prompt_token_ids=token_id, sampling_params=sampling_params
        )

        responses = []
        for output in completions:
            responses.append(output.outputs[0].text)

        res_completions.extend(responses)



    directory = "data/inference"
    file_name = f"{args.model}_{args.size}_{args.data_file}_results.json"
    file_path = os.path.join(directory, file_name)


    if not os.path.exists(directory):

        os.makedirs(directory)


    with open(file_path, "w", encoding="utf8") as f:
        for idx, (instruction, completion) in enumerate(
            tqdm(zip(questions, res_completions))
        ):
            completion_seqs = {"instruction": instruction, "response": completion}
            json.dump(completion_seqs, f, ensure_ascii=False)
            f.write("\n")

    print("Saving results to {}".format(file_path))
    print("Finish")
