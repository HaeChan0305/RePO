import os
import re
import csv
import json
import glob
import random
import argparse
import numpy as np
import pandas as pd
from textwrap import dedent
from pathlib import Path
from typing import List, Dict, Optional

import datasets
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
        
def process_dapo17k(
    max_length: Optional[int] = 1024,
    num_proc: int = 8
) -> DatasetDict:
    """
    Load, preprocess, optionally filter by length, sample, and save DeepMath.

    Args:
        sample_size (int, optional): Number of examples to keep after filtering. 
                                     If None, keep all.
        max_length (int, optional): Maximum allowed token count per example.
                                    If None, skip length filtering.
        num_proc (int): Number of processes for mapping and filtering.

    Returns:
        DatasetDict: The final processed dataset.
    """
    # 1. Load the raw DeepMath dataset
    dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")["train"]
    
    def get_question(example):
        return example["prompt"][0]["content"]
    
    seen_questions = set()
    unique_indices = []
    for idx, example in enumerate(dataset):
        question = get_question(example)
        if question not in seen_questions:
            seen_questions.add(question)
            unique_indices.append(idx)
    
    dataset = dataset.select(unique_indices)
    print(f"After deduplication: {len(dataset)}")
    
    dataset = dataset.select(range(len(dataset) - len(dataset) % 128 + 8))
    print(f"After filtering to 128-batch: {len(dataset)}")

    dataset = DatasetDict({"train": dataset})
    
    # 2. Define preprocessing function
    def _process(item):
        original_opening_instruction = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        original_closing_instruction = '\n\nRemember to put your answer on its own line after \"Answer:\".'
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}.\n\n"

        question = item.pop("prompt")[0]["content"]
        if original_opening_instruction in question and original_closing_instruction in question:
            question = question.replace(original_opening_instruction, "")
            question = question.replace(original_closing_instruction, "")
        else:
            assert False, "Instruction following is not supported"
        
        question = instruction_following + question
        solution = item.pop("reward_model")["ground_truth"]
        

        item["prompt"] = [{"role": "user", "content": question}]
        item["messages"] = [
            {"role": "user", "content": question},
        ]
        item["solution"] = solution
        item["ground_truth"] = solution
        return item

    # 3. Apply mapping and drop unused columns
    keep_cols = ["prompt", "messages", "solution", "ground_truth"]
    drop_cols = [c for c in dataset["train"].column_names if c not in keep_cols]
    dataset = dataset.map(_process, num_proc=num_proc, remove_columns=drop_cols)

    # 4. Initialize tokenizer for length filtering
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

    # 5. Filter by token length if max_length is specified
    if max_length is not None:
        def _filter_by_length(item):
            # Tokenize using the chat template without adding generation prompt
            tokens = tokenizer.apply_chat_template(
                item["messages"], add_generation_prompt=False, tokenize=True
            )
            return len(tokens) < max_length

        dataset = dataset.filter(_filter_by_length, num_proc=num_proc)

    # 7. Print final split sizes
    print("Final dataset splits and sizes:")
    for split, ds in dataset.items():
        print(f"- {split}: {len(ds)} examples")

    # 8. Save to disk with descriptive folder name
    folder_name = f"DAPO-{len(dataset['train'])}"
    save_path = os.path.join("data", folder_name)
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"Saved processed dataset to: {save_path}")

    return dataset

if __name__ == "__main__":
    process_dapo17k(max_length=1024, num_proc=8)
