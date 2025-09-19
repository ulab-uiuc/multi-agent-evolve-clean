#!/usr/bin/env python3
"""
Script to prepare test datasets for evaluation in general tasks.
Supports MATH, HellaSwag, GSM8K, and other common benchmarks.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Any
import argparse
from pathlib import Path


def load_math_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MATH dataset."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['problem']}"}],
            "ground_truth": item['solution'],
            "answer": item['answer'].split('The answer is')[-1].strip() if 'The answer is' in item['answer'] else item['answer'],
            "data_source": "math",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_gsm8k_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Extract the final answer
        answer = item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
        
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['question']}"}],
            "ground_truth": item['answer'],
            "answer": answer,
            "data_source": "gsm8k",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_hellaswag_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load HellaSwag dataset."""
    dataset = load_dataset("Rowan/hellaswag", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = item['endings']
        choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        
        prompt_text = f"""Complete the following scenario by choosing the most likely continuation:

Context: {item['ctx']}

Choices:
{choice_text}

Choose the most appropriate continuation (A, B, C, or D):"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": chr(65 + int(item['label'])),  # Convert 0,1,2,3 to A,B,C,D
            "answer": chr(65 + int(item['label'])),
            "data_source": "hellaswag",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
    
    return data


def load_mmlu_dataset(subject: str = "abstract_algebra", split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MMLU dataset for a specific subject."""
    dataset = load_dataset("cais/mmlu", subject, split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = item['choices']
        choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        
        prompt_text = f"""Answer the following multiple choice question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": chr(65 + item['answer']),  # Convert 0,1,2,3 to A,B,C,D
            "answer": chr(65 + item['answer']),
            "data_source": f"mmlu_{subject}",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
        print(f"Loaded {subject} sample {i+1}: {item['question']}")
        print(f"Choices: {choice_text}")
        print(f"Ground truth: {data[-1]['ground_truth']}")
        print(f"Answer: {data[-1]['answer']}")
        print("-" * 40)
    return data


def load_arc_dataset(split: str = "test", challenge: bool = True, num_samples: int = None) -> List[Dict]:
    """Load ARC dataset."""
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = [choice for choice in item['choices']['text']]
        labels = item['choices']['label']
        choice_text = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
        
        prompt_text = f"""Answer the following science question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer:"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": item['answerKey'],
            "answer": item['answerKey'],
            "data_source": f"arc_{'challenge' if challenge else 'easy'}",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
    
    return data

def load_aime24_dataset(split: str = "train", num_samples: int = None) -> List[Dict]:
    """Load AIME 2024 dataset."""
    dataset = load_dataset("HuggingFaceH4/aime_2024", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['problem']}"}],
            "ground_truth": item['answer'],
            "answer": item['answer'],
            "data_source": "aime24",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_truthfulqa_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load TruthfulQA dataset."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Answer the following question truthfully:\n\n{item['question']}"}],
            "ground_truth": item['best_answer'],
            "answer": item['best_answer'],
            "data_source": "truthfulqa",
            "extra_info": {"metric": "truthfulness_accuracy"}
        })
    
    return data

def load_gpqa_dataset(split: str = "train", num_samples: int = None) -> List[Dict]:
    """Load GPQA dataset."""
    dataset = load_dataset("Idavidrein/gpqa", 'gpqa_main', split=split)
    
    data = []
    for i, item in enumerate(dataset):
        # if num_samples and i >= num_samples:
        #     break
            
        # Format the question and choices
        question = item['Question']
        correct_answer = item['Correct Answer']
        incorrect_answers = [
            item['Incorrect Answer 1'],
            item['Incorrect Answer 2'], 
            item['Incorrect Answer 3']
        ]
        
        # Combine all answers and shuffle them
        all_answers = [correct_answer] + incorrect_answers
        import random
        random.shuffle(all_answers)
        
        # Find which position the correct answer is in
        correct_index = all_answers.index(correct_answer)
        correct_letter = chr(65 + correct_index)  # Convert to A, B, C, D
        
        # Format choices
        choice_text = "\n".join([f"{chr(65+j)}. {answer}" for j, answer in enumerate(all_answers)])
        
        prompt_text = f"""Answer the following scientific question:

Question: {question}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
        
        # Get additional metadata
        subdomain = item.get('Subdomain', '')
        high_level_domain = item.get('High-level domain', '')
        difficulty = item.get("Writer's Difficulty Estimate", '')
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": correct_answer,
            "answer": correct_letter,
            "data_source": "gpqa",
            "extra_info": {
                "metric": "multiple_choice_accuracy",
            }
        })
        
        print(f"Loaded GPQA sample {i+1}: {question[:100]}...")
        print(f"Answer: {data[-1]['answer']}")
    
    return data

def load_mmlu_dataset_as_a_whole(split: str = "test", num_samples: int = None) -> List[Dict]:
    # enumerate all subjects in MMLU
    MMLU_SUBJECTS = [
        "abstract_algebra",
        "anatomy", 
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry", 
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics", 
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law", 
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]
    
    data = []
    total_collected = 0
    
    for subject in MMLU_SUBJECTS:
        if num_samples and total_collected >= num_samples:
            break
            
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)
            
            subject_samples = 0
            for i, item in enumerate(dataset):
                if num_samples and total_collected >= num_samples:
                    break
                    
                choices = item['choices']
                choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
                
                prompt_text = f"""Answer the following multiple choice question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
                
                data.append({
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "ground_truth": chr(65 + item['answer']),  # Convert 0,1,2,3 to A,B,C,D
                    "answer": chr(65 + item['answer']),
                    "data_source": f"mmlu",
                    "extra_info": {
                        "metric": "multiple_choice_accuracy",
                    }
                })
                
                total_collected += 1
                subject_samples += 1
                
            print(f"Loaded {subject_samples} samples from MMLU subject: {subject}")
            
        except Exception as e:
            print(f"Failed to load MMLU subject {subject}: {e}")
            continue
    
    print(f"Total MMLU samples loaded: {len(data)}")
    return data


def load_bbh_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load BBH dataset."""
    BBH_SUBSETS = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "movie_recommendation",
        'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'
    ]
    data = []
    for subset in BBH_SUBSETS:
        # dataset = load_dataset("SaylorTwift/bbh", split=split)
        dataset = load_dataset("SaylorTwift/bbh", subset, split=split)
        for i, item in enumerate(dataset):
            # if num_samples and i >= num_samples:
            #     break
            question = item['input']
            correct_answer = item['target']

            data.append({
                "prompt": [{"role": "user", "content": question}],
                "ground_truth": correct_answer,
                "answer": correct_answer,
                "data_source": "bbh",
                "extra_info": {
                    "metric": "multiple_choice_accuracy",
                }
            })

            print(f"Loaded BBH sample {i+1}: {question[:100]}...")
            # print(f"Loaded BBH sample {i+1}: {question[:100]}...")
            # print(f"Choices: {choice_text}")
            print(f"Ground truth: {data[-1]['ground_truth']}")
            print(f"Answer: {data[-1]['answer']}")
            print("-" * 40)

    return data

def save_dataset_to_parquet(data: List[Dict], output_path: str, dataset_name: str):
    """Save dataset to parquet format."""
    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(data)
    output_file = os.path.join(output_path, f"{dataset_name}_test.parquet")
    df.to_parquet(output_file)
    print(f"Saved {len(data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare test datasets for evaluation")
    parser.add_argument("--output_dir", type=str, default="./validation_datasets", 
                       help="Output directory for validation datasets")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to load per dataset (None for all)")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["math", "gsm8k", "hellaswag", "mmlu", "arc", "truthfulqa", "aime24", "gpqa", "all"],
                       default=["all"], help="Datasets to prepare")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    datasets_to_load = args.datasets
    if "all" in datasets_to_load:
        datasets_to_load = ["math", "gsm8k", "hellaswag", "mmlu", "arc", "truthfulqa", "aime24", "gpqa", "bbh"]
    
    print(f"Preparing datasets: {datasets_to_load}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples per dataset: {args.num_samples or 'All'}")
    
    # Load and save datasets
    if "math" in datasets_to_load:
        print("\nLoading MATH dataset...")
        math_data = load_math_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(math_data, args.output_dir, "math")
    
    if "gsm8k" in datasets_to_load:
        print("\nLoading GSM8K dataset...")
        gsm8k_data = load_gsm8k_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(gsm8k_data, args.output_dir, "gsm8k")
    
    # if "hellaswag" in datasets_to_load:
    #     print("\nLoading HellaSwag dataset...")
    #     hellaswag_data = load_hellaswag_dataset(num_samples=args.num_samples)
    #     save_dataset_to_parquet(hellaswag_data, args.output_dir, "hellaswag")
    
    if "mmlu" in datasets_to_load:
        print("\nLoading MMLU dataset (all subjects)...")
        
        # First get the MMLU_SUBJECTS list from the function
        def get_mmlu_subjects():
            return [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
                "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
                "college_medicine", "college_physics", "computer_security", "conceptual_physics",
                "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
                "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
                "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
                "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
                "high_school_physics", "high_school_psychology", "high_school_statistics",
                "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
                "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
                "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
                "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
                "professional_law", "professional_medicine", "professional_psychology", "public_relations",
                "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
            ]
        
        subjects = get_mmlu_subjects()
        for subject in subjects:
            mmlu_data = load_mmlu_dataset(subject=subject, num_samples=args.num_samples)
            save_dataset_to_parquet(mmlu_data, args.output_dir, f"mmlu_{subject}")
        mmlu_data = load_mmlu_dataset_as_a_whole(num_samples=args.num_samples)
        save_dataset_to_parquet(mmlu_data, args.output_dir, f"mmlu")
    
    if "arc" in datasets_to_load:
        print("\nLoading ARC dataset...")
        arc_challenge_data = load_arc_dataset(challenge=True, num_samples=args.num_samples)
        save_dataset_to_parquet(arc_challenge_data, args.output_dir, "arc_challenge")
        
        arc_easy_data = load_arc_dataset(challenge=False, num_samples=args.num_samples)
        save_dataset_to_parquet(arc_easy_data, args.output_dir, "arc_easy")
    
    if "truthfulqa" in datasets_to_load:
        print("\nLoading TruthfulQA dataset...")
        truthfulqa_data = load_truthfulqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(truthfulqa_data, args.output_dir, "truthfulqa")

    if "aime24" in datasets_to_load:
        print("\nLoading AIME 2024 dataset...")
        aime24_data = load_aime24_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(aime24_data, args.output_dir, "aime24")

    if "gpqa" in datasets_to_load:
        print("\nLoading GPQA dataset...")
        gpqa_data = load_gpqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(gpqa_data, args.output_dir, "gpqa")

    if "bbh" in datasets_to_load:
        print("\nLoading BBH dataset...")
        bbh_data = load_bbh_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(bbh_data, args.output_dir, "bbh")

    print(f"\nAll datasets prepared and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
