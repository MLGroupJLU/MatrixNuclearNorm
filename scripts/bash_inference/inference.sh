#!/bin/bash


models=("Llama-2")
sizes=("70b")

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        
      echo "Running with model=$model, size=$size"
      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "alpaca"

      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "arena"     
    done
done

models=("Llama-3")
sizes=("8B" "70B")

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        
      echo "Running with model=$model, size=$size"
      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "alpaca"

      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "arena"     
    done
done

models=("qwen2")
sizes=("0.5B" "1.5B" "7B") # 72B

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        
      echo "Running with model=$model, size=$size"
      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "alpaca"

      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "arena"     
    done
done

models=("qwen1.5")
sizes=("0.5B" "1.8B" "4B" "7B" "14B") # 72B

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        
      echo "Running with model=$model, size=$size"
      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "alpaca"

      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "arena"     
    done
done

models=("vicuna")
sizes=("7b" "13b" "33B") #

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        
      echo "Running with model=$model, size=$size"
      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "alpaca"

      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "arena"     
    done
done

models=("deepseek")
sizes=("1.3b" "6.7b" "7b")

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        
      echo "Running with model=$model, size=$size"
      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "alpaca"

      CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/vllm_inference.py \
          --model "$model" \
          --size "$size" \
          --data_file "arena"     
    done
done

CUDA_VISIBLE_DEVICES=0 python ../../Experiment-mnn/code/vllm_inference.py \
    --model "gemma" \
    --size "7b" \
    --data_file "alpaca"

CUDA_VISIBLE_DEVICES=0 python ../../Experiment-mnn/code/vllm_inference.py \
    --model "gemma" \
    --size "7b" \
    --data_file "arena"

CUDA_VISIBLE_DEVICES=0 python ../../Experiment-mnn/code/vllm_inference.py \
    --model "mistral" \
    --size "7B" \
    --data_file "alpaca"

CUDA_VISIBLE_DEVICES=0 python ../../Experiment-mnn/code/vllm_inference.py \
    --model "mistral" \
    --size "7B" \
    --data_file "arena"
