#!/bin/bash


models=("gpt")
sizes=("111M" "256M" "590M" "1.3B" "2.7B" "6.7B" "13B")

datasets=("dolly" "rlhf" "openwebtext2" "wiki" "openbookqa" "piqa" "winogrande")

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            echo "Running with model=$model, size=$size, dataset=$dataset"
            CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/compute_mnn_time.py \
                --model "$model" \
                --size "$size" \
                --dataset "$dataset"
        done
    done
done

models=("pythia")
sizes=("14m" "31m" "70m" "160m" "410m" "1B" "1.4B" "2.8B" "6.9B" "12B")
datasets=("openwebtext2")

for model in "${models[@]}"
do
    for size in "${sizes[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            echo "Running with model=$model, size=$size, dataset=$dataset"
            CUDA_VISIBLE_DEVICES=0,1 python ../../Experiment-mnn/code/compute_mnn_time.py \
                --model "$model" \
                --size "$size" \
                --dataset "$dataset"
        done
    done
done



