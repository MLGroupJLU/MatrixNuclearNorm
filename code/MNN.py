import sys
import os
import time
import torch
import transformers
import json
import jsonlines
import argparse
import numpy as np
import heapq
import math
from tqdm import tqdm
import torch.distributed as dist
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM

MAX_INT = sys.maxsize

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R


def matrix_nuclear_norm(X, D=None):
    if D is None:
        D = min(X.shape[0], X.shape[1])

    l2_norms = torch.sqrt(torch.sum(torch.pow(X, 2), dim=0))
    list_svd, _ = torch.sort(l2_norms, descending=True)
    top_D_l2_norms = list_svd[:D]
    L_MNN = torch.sum(top_D_l2_norms)
    MNN_normalize = L_MNN.item()
    return MNN_normalize


def normalize_and_compute_matrix_nuclear_norm(R, D=None):

    R_normalized = normalize(R)


    MNN_normalize = matrix_nuclear_norm(R_normalized, D)

    return MNN_normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--size", type=str, default="2.7B")
    parser.add_argument("--data", type=str, default="alpaca")
    parser.add_argument("--start_idx", type=int, default=0)  # start index
    parser.add_argument("--end_idx", type=int, default=9999999)  # end index
    parser.add_argument("--file_path", type=str, default="") 

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading model ...")
    args = parse_args()
    args = parse_args()
    if args.model == "gpt":
        model_path = f"../../cerebras/Cerebras-GPT-{args.size}"  # 111M 256M 590M 1.3B 2.7B 6.7B 13B
    elif args.model == "pythia":
        model_path = f"../../EleutherAI/pythia-{args.size}"  # 14m 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b
    elif args.model == "Llama-2":  # -7b-chat-hf
        model_path = f"../../meta-llama/Llama-2-{args.size}-chat-hf"  # 7b 13b
    elif args.model == "Llama-3":  # -7b-chat-hf
        model_path = f"../../meta-llama/Meta-Llama-3-{args.size}-Instruct"  # 7b 13b
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

    if "111M" in model_path or "31m" in model_path:
        model = AutoModel.from_pretrained(model_path, device_map="auto").eval().to(device)        
    else:
        model = AutoModel.from_pretrained(model_path,torch_dtype="auto",device_map="auto")   

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Processing data ...")
    instructions, outputs = [], []

    if "wizard" in args.model.lower():
        model_name = "wizard"
    elif "gemma" in args.model.lower():
        model_name = "gemma"
    elif "gpt" in args.model.lower():
        model_name = "gpt"
    elif "pythia" in args.model.lower():
        model_name = "pythia"
    elif "deepseek" in args.model.lower():
        model_name = "deepseek"
    elif "qwen" in args.model.lower():
        model_name = "qwen"
    elif "llama" in args.model.lower():
        model_name = "llama"
    elif "vicuna" in args.model.lower():
        model_name = "vicuna"
    elif "mistral" in args.model.lower():
        model_name = "mistral"

    start_time = time.time()
    data = []

    directory = f"{args.file_path}/data/inference"
    file_name = f"{args.model}_{args.size}_{args.data}_results.json"
    data_path = os.path.join(directory, file_name)

    with open(data_path, "r", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            data.append(item)

    for item in data:
        instructions.append(item["instruction"])
        outputs.append(item["response"])

    instructions = instructions[args.start_idx : args.end_idx]
    outputs = outputs[args.start_idx : args.end_idx]

    prompt_instruction = {
        "gpt": "USER:\n{instruction}\nASSISTANT:\n",
        "pythia": "USER:\n{instruction}\nASSISTANT:\n",
        "gemma": "USER:\n{instruction}\nASSISTANT:\n",
        "deepseek": "USER:\n{instruction}\nASSISTANT:\n",
        "qwen": "USER:\n{instruction}\nASSISTANT:\n",
        "wizard": "USER:\n{instruction}\nASSISTANT:\n",
        "llama": "USER:\n{instruction}\nASSISTANT:\n",
        "vicuna": "USER:\n{instruction}\nASSISTANT:\n",  
        "mistral": "USER:\n{instruction}\nASSISTANT:\n",  

    }

    response_mnn = []
    for i in tqdm(range(len(instructions))):

        messages = outputs[i]
        # messages = messages + outputs[i]
        tokenized = tokenizer(messages, return_tensors="pt").to(device)
        token_ids = tokenized["input_ids"].to(device)

        with torch.no_grad():
            R = model(token_ids)[0][0, :, :]
            if R.shape[0] == 0:
                response_mnn.append(-1)
            else:
                mnn = normalize_and_compute_matrix_nuclear_norm(R)
                response_mnn.append(mnn / len(token_ids[0]))

    end_time = time.time()
    runtime = end_time - start_time

    data_list = []
    data_list.append({"runtime": str(runtime)})
    for i in range(len(instructions)):
        tmp = {}
        tmp["id"] = i + args.start_idx
        tmp["instruction"] = instructions[i]
        tmp["output"] = outputs[i]
        tmp["response_mnn"] = response_mnn[i]
        data_list.append(tmp)

    directory = f"{args.file_path}/data/result_MNN"
    file_name = f"{args.model}_{args.size}_{args.data}_results.json"
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "w+", encoding="utf8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
