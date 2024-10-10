import sys
import os
import json
from tqdm import tqdm
import argparse
import math

MAX_INT = sys.maxsize

if __name__ == "__main__":
    path = "./Experiment-mnn/data/result_MNN/"
    all_results = []

    for sub_dir in os.listdir(path):
        sub_dir_path = os.path.join(path, sub_dir)
        data = []
        try:
            with open(sub_dir_path, "r") as f:
                data = json.load(f)

            mnn = [item["response_mnn"] for item in data if "response_mnn" in item]
            valid_mnn = [e for e in mnn if not math.isnan(e)]
            avg_mnn = sum(valid_mnn) / len(valid_mnn) if valid_mnn else None

            # 提取 runtime 值
            runtime = data[0]["runtime"] if data and "runtime" in data[0] else None

            all_results.append(
                {"sub_dir": sub_dir, "average_MNN": avg_mnn, "runtime": runtime}
            )
            print(sub_dir_path)
            if avg_mnn is not None:
                print(f"Average Entropy: {avg_mnn}")
            if runtime is not None:
                print(f"Runtime: {runtime}")
        except Exception as e:
            print(f"Error processing {sub_dir_path}: {e}")
            all_results.append(
                {"sub_dir": sub_dir, "average_mnn": None, "runtime": None}
            )

    # 对结果按照子目录名称排序
    results = sorted(all_results, key=lambda x: x["sub_dir"])

data_dir = os.path.join("./Experiment-mnn/data/output_MNN")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 指定结果文件的路径
results_file_path = os.path.join(data_dir, "Average_MNN_results_with_runtime.jsonl")

# 将排序后的结果写入文件
with open(results_file_path, "w") as f:
    for result in results:
        json.dump(result, f)
        f.write("\n")
