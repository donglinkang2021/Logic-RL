""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset
from tqdm import tqdm
import argparse
import json
from pathlib import Path

def make_prefix(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

def make_map_fn(split, template_type):
    def process_fn(example, idx):
        question = make_prefix(example, template_type=template_type)
        solution = {
            "solution_text_format": example['solution_text_format'],
            "statements": example['statements']
        }
        data = {
            "data_source": 'kk_logic',
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "logic",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data
    return process_fn

def gen_from_jsonl(path):
    """Load custom JSONL dataset"""
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def main():
    template_type_list = ['base', 'qwen-instruct']

    data_class_list = [
      "clean", 
      "flip_role", 
      "perturbed_leaf", 
      "perturbed_statement", 
      "random_pair", 
      "reorder_statement", 
      "uncommon_name"  
    ]

    for template_type in template_type_list:
        for data_class in data_class_list:

            save_dir = f"./data/kk/{data_class}/{template_type}/"
            data_path = f'./kk_data/data/train/{data_class}'
            
            # 使用pathlib获取所有jsonl文件
            jsonl_files = [f.resolve() for f in Path(data_path).glob('*.jsonl')]
            
            for jsonl_file in jsonl_files:
                file_name = jsonl_file.stem
                people, numbers = file_name.split('_')
                people_num = int(people.split('people')[1])
                num_samples = int(numbers.split('num')[1])

                jsonl_file = str(jsonl_file)

                raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': jsonl_file})
                print(len(raw_dataset))

                train_size = int(0.9 * len(raw_dataset))
                train_dataset = raw_dataset.select(range(train_size))
                test_dataset = raw_dataset.select(range(train_size, len(raw_dataset)))

                train_dataset = train_dataset.map(function=make_map_fn('train', template_type), with_indices=True)
                test_dataset = test_dataset.map(function=make_map_fn('test', template_type), with_indices=True)

                # Create local directory if not exists
                save_dataset_dir = Path(save_dir) / f'num{num_samples}' / f'{people_num}ppl'
                save_dataset_dir.mkdir(parents=True, exist_ok=True)

                train_dataset.to_parquet(save_dataset_dir / 'train.parquet')
                test_dataset.to_parquet(save_dataset_dir / 'test.parquet')


if __name__ == '__main__':
    main()
    
