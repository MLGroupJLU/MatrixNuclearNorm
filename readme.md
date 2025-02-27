# Large Language Model Evaluation via Matrix Nuclear Norm

<p align="center">
    🤗 <a href="https://huggingface.co/papers/2410.10672" target="_blank">Hugging Face</a> • ⏬ <a href="#Datasets" target="_blank">Data</a> •  📃 <a href="https://arxiv.org/pdf/2410.10672" target="_blank">Paper</a>
</p>    

## Introduction

As large language models (LLMs) continue to evolve, efficient evaluation metrics are vital for assessing their ability to compress information and reduce redundancy. While traditional metrics like Matrix Entropy offer valuable insights, they are computationally intensive for large-scale models due to their \( O(n^3) \) time complexity with Singular Value Decomposition (SVD). To mitigate this issue, we introduce the Matrix Nuclear-Norm, which not only serves as a metric to quantify the data compression proficiency of LLM but also provides a convex approximation of matrix rank to capture both predictive discriminability and diversity. By employing the \( L_{1,2}\text{-norm} \) to further approximate the nuclear norm, we can effectively assess the model's information compression capabilities. This approach reduces the time complexity to \( O(n^2) \) and eliminates the need for SVD computation. Consequently, the Matrix Nuclear-Norm achieves speeds 8 to 24 times faster than Matrix Entropy for the CEREBRAS-GPT model as sizes increase from 111M to 6.7B. This performance gap becomes more pronounced with larger models, as validated in tests with other models like Pythia. Additionally, evaluations on benchmarks and model responses confirm that our proposed Matrix Nuclear-Norm is a reliable, scalable, and efficient tool for assessing LLMs' performance, striking a balance between accuracy and computational efficiency.

<div align="center">
    <img src="./Figures/time-gpt.png" alt="Alt Text" width="600"/>
</div>


## Calculation of Matrix Nuclear Norm
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Normalize the input tensor R (N*d matrix)
def normalize(R):
    with torch.no_grad():
        R = R - R.mean(dim=0)  
        norms = torch.norm(R, p=2, dim=1, keepdim=True) 
        R = R / norms 
    return R

# Compute the Matrix Nuclear Norm (MNN) for a matrix X, limited by dimension D
def matrix_nuclear_norm(X, D=None):
    if D is None:
        D = min(X.shape[0], X.shape[1])  

    l2_norms = torch.sqrt(torch.sum(X ** 2, dim=0))  
    sorted_norms, _ = torch.sort(l2_norms, descending=True) 
    top_D_norms = sorted_norms[:D] 
    MNN = torch.sum(top_D_norms).item() 
    return MNN

# Load the model and tokenizer
model_path = "cerebras/Cerebras-GPT-1.3B"  # Example model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, device_map="auto").cuda()

# Example text input
text = "The sky is a lovely shade of blue today."
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# Compute Matrix Nuclear Norm for the model's output
with torch.no_grad():
    R = model(inputs.input_ids)[0][0, :, :]  # Extract hidden states from the model
    R = normalize(R)  # Normalize the hidden states
    MNN = matrix_nuclear_norm(R)  # Compute the Matrix Nuclear Norm

print(f"Matrix Nuclear Norm: {MNN}")

```


## Datasets

- **Validation Datasets**  
  Please download the validation datasets from [Wiki-en](https://huggingface.co/datasets/wikipedia), [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [OpenWebText2](https://huggingface.co/datasets/suolyer/pile_openwebtext2), and [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) and update the data paths in your scripts accordingly.

- **Inference and Evaluation Datasets**  
  For inference and evaluation, download the datasets from [Arena-Hard](https://huggingface.co/datasets/pvduy/arena-hard) and [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca).

- **Prompt Test Datasets**  
  Download the prompt test dataset from [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca).


## Experiments:

### Matrix Nuclear Norm of Dataset

#### 1. Compute Time

To compare the computation times, navigate to the `scripts/bash_final` directory and execute the following shell scripts:

```bash
cd scripts/bash_final
bash ./time-entropy.sh
bash ./time-mnn.sh
```

#### 2. Run Inference for Selected Models

To perform inference on a subset of models, change to the `scripts/bash_inference` directory and run the following script:

```bash
cd scripts/bash_inference
bash ./inference.sh
```

#### 3. Calculate Matrix Nuclear Norm (MNN) Results

To calculate the MNN results, execute the script located in the `bash_MNN` directory:

```bash
bash ./bash_MNN/compute_mnn.sh
```

#### 4. Compute Scores

Finally, to compute the scores, run the Python script as follows:

```python
python ./path/Experiment-mnn/code/compute_score.py
```

## Citation
If you are utilizing Matrix Nuclear norm in your research or applications, please reference it using the following BibTeX entry:
<p>
    <a href="https://huggingface.co/papers/2410.10672">
        <img class="blinking" src="https://img.shields.io/badge/Project-Link-Green" alt="Project Link">
    </a>    
    <a href="https://arxiv.org/pdf/2410.10672">
        <img class="blinking" src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Arxiv Paper">
    </a>
</p>

```
@article{li2024large,
  title={Large Language Model Evaluation via Matrix Nuclear-Norm},
  author={Li, Yahan and Xia, Tingyu and Chang, Yi and Wu, Yuan},
  journal={arXiv preprint arXiv:2410.10672},
  year={2024}
}
```
