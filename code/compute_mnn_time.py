import csv
import os
import time
import numpy as np
from regex import F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Model, AutoModel
import torch
import math
import json
import tqdm
import random
from datasets import load_dataset, load_from_disk


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


def normalize_and_compute_fast_nuclear_norm(R, D=None):

    R_normalized = normalize(R)

    MNN_normalize = matrix_nuclear_norm(R_normalized, D)

    return MNN_normalize

def jsonl_to_list(filename):
    data_list = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data['chosen'])

    return data_list

def main(args):

    start_time = time.time()
    if args.model == "gpt":
        model_path = f"{args.file_path}/model/cerebras/Cerebras-GPT-{args.size}"  # 111M 256M 590M 1.3B 2.7B 6.7B 13B
    elif args.model == "pythia":
        model_path = f"{args.file_path}/model/EleutherAI/pythia-{args.size}"  # 14m 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b
    elif args.model == "Llama-2":  # -7b-chat-hf
        model_path = f"{args.file_path}/model/meta-llama/Llama-2-{args.size}-chat-hf"  # 7b 13b
    elif args.model == "Llama-3":  # -7b-chat-hf
        model_path = f"{args.file_path}/model/meta-llama/Meta-Llama-3-{args.size}-Instruct"  # 7b 13b
    elif args.model == "deepseek":
        if args.size == "7b":
            model_path = f"{args.file_path}/model/deepseek-ai/deepseek-coder-{args.size}-instruct-v1.5"
        else:
            model_path = f"{args.file_path}/model/deepseek-ai/deepseek-coder-{args.size}-instruct"
    elif args.model == "qwen2":
        model_path = f"{args.file_path}/model/Qwen/Qwen2-{args.size}-Instruct"
    elif args.model == "qwen1.5":
        model_path = f"{args.file_path}/model/Qwen/Qwen1.5-{args.size}-Chat"
    elif args.model == "mistral":
        model_path = f"{args.file_path}/model/mistralai/Mistral-{args.size}-Instruct-v0.3"
    elif args.model == "gemma":
        model_path = f"{args.file_path}/model/google/gemma-{args.size}"
    elif args.model == "WizardLMTeam":
        if args.size == "13B":
            model_path = f"{args.file_path}/model/WizardLMTeam/WizardLM-{args.size}-V1.2"
        else:
            model_path = f"{args.file_path}/model/WizardLMTeam/WizardLM-{args.size}-V1.0"
    elif args.model == "vicuna":
        if args.size == "33B":
            model_path = f"{args.file_path}/model/lmsys/vicuna-{args.size}-v1.3"
        else:
            model_path = f"{args.file_path}/model/lmsys/vicuna-{args.size}-v1.5"
    else:
        pass
      
    if "111M" in model_path or "31m" in model_path:
          model = AutoModel.from_pretrained(model_path, device_map="auto").eval().to(device)        
    else:
          model = AutoModel.from_pretrained(model_path,torch_dtype="auto",device_map="auto")
          
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_ids = []

    if args.dataset == "dolly":
        with open('../Experiment-mnn/data/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl', 'r') as file:
            for line in file:
                json_line = json.loads(line)
                context = json_line.get('context', '')  

                if len(context)>0:
                    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)                
                    input_ids.append(inputs.input_ids)

    elif args.dataset == "wiki":
        dataset = load_from_disk("../Experiment-mnn/data/wikipedia")
        sample_size = 10000
        # sample_size = 10000
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset['train'])
        random_indices = random.sample(range(dataset_size), sample_size)

        for i, idx in tqdm.tqdm(enumerate(random_indices)):
            sample = dataset['train'][idx]
            context = sample['text']
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)

            input_ids.append(inputs.input_ids)

    elif args.dataset == "rlhf":
        ref_list1 = jsonl_to_list('./Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/harmless-base/test.jsonl')
        ref_list2 = jsonl_to_list('./Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-base/test.jsonl')
        ref_list3 = jsonl_to_list('./Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-online/test.jsonl')
        ref_list4 = jsonl_to_list('./Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-rejection-sampled/test.jsonl')
        dataset = ref_list1 + ref_list2 + ref_list3 + ref_list4
        for context in tqdm.tqdm(dataset):
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)              
            input_ids.append(inputs.input_ids)
    elif args.dataset == "openbookqa":
        dataset = load_from_disk(f"{args.file_path}/data/allenai-openbookqa/train") 
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset)
        random_indices = random.sample(range(dataset_size), sample_size)
        for i, idx in enumerate(random_indices):
            context = " question: "+dataset['question_stem'][idx] + " " \
                      + dataset['choices'][idx]['label'][0]+ ": " + dataset['choices'][idx]['text'][0]+ " "  \
                      + dataset['choices'][idx]['label'][1]+ ": " + dataset['choices'][idx]['text'][1]+ " "  \
                      + dataset['choices'][idx]['label'][2]+ ": " + dataset['choices'][idx]['text'][2]+ " "  \
                      + dataset['choices'][idx]['label'][3]+ ": " + dataset['choices'][idx]['text'][3]+ " "  \
                      +" answerKey "+ dataset['answerKey'][idx]
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)        
            input_ids.append(inputs.input_ids) 

    elif args.dataset == "piqa":
        dataset = load_from_disk(f"{args.file_path}/data/piqa/train")  
        sample_size = 1000
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset)
        random_indices = random.sample(range(dataset_size), sample_size)
        for i, idx in enumerate(random_indices):
            context = "goal: "+ dataset['goal'][idx] +" solution1: "+ dataset['sol1'][idx] +" solution2: "+ dataset['sol2'][idx] +" label: "+ str(dataset['label'][idx])
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)        
            input_ids.append(inputs.input_ids)  

    elif args.dataset == "winogrande":
        dataset = load_from_disk(f"{args.file_path}/data/winogrande_m/train") 
        sample_size = 1000
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset)
        random_indices = random.sample(range(dataset_size), sample_size)
        for i, idx in enumerate(random_indices):
            context = "sentece: "+dataset['sentence'][idx] +" option1: "+ dataset['option1'][idx] +" option2: "+ dataset['option2'][idx] +" answer: "+ dataset['answer'][idx]
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)        
            input_ids.append(inputs.input_ids)  

    ls1 = []
    with tqdm.tqdm(input_ids, desc="MNN: - ") as progress:
        for id in progress:
            with torch.no_grad():

                r = model(id.to(device))[0][0, :, :]

                res_MNN= normalize_and_compute_fast_nuclear_norm(r)
                ls1.append(res_MNN/len(id(0)))

            torch.cuda.empty_cache()
            progress.set_description(f"MNN: {res_MNN:.4f}")

    res_MNN_avg = np.nansum(ls1) / np.sum(~np.isnan(ls1))

    result_dir = 'result_final_Time'
    file_path = os.path.join(result_dir, f'Time_{args.dataset}_MNN_{args.model}_results.csv')

    # 确保目录存在
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 定义字段名
    fieldnames = ['Run Time', 'MNN']

    # 检查文件是否存在，如果不存在，写入表头
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 写入数据
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        run_time = time.time() - start_time  # 计算运行时间
        writer.writerow({'Run Time': f"{run_time}s", 'MNN': f"{args.model}-{args.size}-MNN: {res_MNN_avg}"}) 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--size", type=str, default="111M")
    parser.add_argument("--dataset", type=str, default="openwebtext2") # dolly  wiki openwebtext2 rlhf
    parser.add_argument("--file_path", type=str, default="./Experiment-mnn") # dolly  wiki openwebtext2 rlhf
    args = parser.parse_args()
    
    main(args)
