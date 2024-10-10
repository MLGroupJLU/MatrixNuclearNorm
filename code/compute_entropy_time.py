import csv
import os
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Model, AutoModel
import torch
import math
import json
import tqdm
import random
from datasets import load_dataset, load_from_disk
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
    
    
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R

def cal_cov(R):
    with torch.no_grad():
        Z = torch.nn.functional.normalize(R, dim=1)
        A = torch.matmul(Z.T, Z)/Z.shape[0]
    return A

def cal_entropy(A):
    with torch.no_grad():
        eig_val = torch.svd(A / torch.trace(A))[1] 
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        normalized_entropy = entropy/math.log(A.shape[0])
    return normalized_entropy


def jsonl_to_list(filename):
    data_list = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data['chosen'])

    return data_list
  

def main(args):
  
  
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass
        
    model_path = ''
    model = ''    
    start_time = time.time()
    if args.model == "gpt":
        model_path = f"Experiment-mnn/model/cerebras/Cerebras-GPT-{args.size}" # 111M 256M 590M 1.3B 2.7B 6.7B 13B
    elif args.model == "pythia":
        model_path = f"Experiment-mnn/model/EleutherAI/pythia-{args.size}" # 14m 70m 160m 410m 1b 1.4b 2.8b 6.9b 12b
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
        ref_list1 = jsonl_to_list('../Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/harmless-base/test.jsonl')
        ref_list2 = jsonl_to_list('../Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-base/test.jsonl')
        ref_list3 = jsonl_to_list('../Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-online/test.jsonl')
        ref_list4 = jsonl_to_list('../Experiment-mnn/data/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-rejection-sampled/test.jsonl')
        dataset = ref_list1 + ref_list2 + ref_list3 + ref_list4
        for context in tqdm.tqdm(dataset):
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)              
            input_ids.append(inputs.input_ids)
    
    elif args.dataset == "openwebtext2":
        dataset = load_from_disk("../Experiment-mnn/data/suolyer/pile_openwebtext2/validation")
        sample_size = 10000
        random_seed = 123
        random.seed(random_seed)
        dataset_size = len(dataset)
        random_indices = random.sample(range(dataset_size), sample_size)
        for i, idx in enumerate(random_indices):
            context = dataset[idx]['text']
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)        
            input_ids.append(inputs.input_ids)
            
  
            
    Entropy1 = 0.0
    ls1, ls2, ls3 = [], [], []
    with tqdm.tqdm(input_ids, desc="Entropy: - ") as progress:
        for id in progress:
            with torch.no_grad():
                # 计算entropy
                
                r = model(id.to(device))[0][0, :, :]
                R = normalize(r)
                A = cal_cov(R)
                Entropy1 = cal_entropy(A)
                
                ls1.append(Entropy1)
                
            torch.cuda.empty_cache()
            progress.set_description(f"Entropy: {Entropy1:.4f}")
            
    
    entropy_avg1 = np.nansum(ls1) / np.sum(~np.isnan(ls1))

     
    result_dir = 'result_final_Time'
    file_path = os.path.join(result_dir, f'Time_{args.dataset}_entropy_{args.model}_results.csv')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fieldnames = ['Run Time', 'entropy']

    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        run_time = time.time() - start_time  # 计算运行时间
        writer.writerow({'Run Time': f"{run_time}s", 'entropy': f"{args.model}-{args.size}-entropy: {entropy_avg1}"}) 
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--size", type=str, default="1.3B")
    parser.add_argument("--dataset", type=str, default="dolly") 
    args = parser.parse_args()
    
    main(args)