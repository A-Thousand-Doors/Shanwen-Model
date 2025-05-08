import argparse
import os
import re

import datasets

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """
    Extract the solution from the given solution string.
    Args:
        solution_str (str): The solution string to be processed.
    Returns:
        str: The extracted solution.
    """
    solution = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if solution:
        return solution.group(1).strip().replace(",", "").replace("$", "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")

    args = parser.parse_args()

    dataset = pd.read_parquet(args.data_path)

    total = len(dataset)
    correct = 0

    for _, row in dataset.iterrows():
        answer = extract_solution(row["responses"])
        gt = row["reward_model"]["ground_truth"]
        if answer == gt:
            correct += 1
    
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total:.4%}")
