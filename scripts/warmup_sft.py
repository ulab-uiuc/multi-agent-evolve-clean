#!/usr/bin/env python3
"""
Warm-up SFT script using VERL's FSDP SFT Trainer
This script performs supervised fine-tuning on curated data as a warm-up phase
before PPO training to ensure stable output format.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional

# Add VERL to path if needed
sys.path.insert(0, '/home/yidingw/miniconda3/envs/mae/lib/python3.10/site-packages')

from verl.trainer.fsdp_sft_trainer import run_sft, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


def prepare_warmup_data(
    data_path: str,
    output_path: str,
    max_samples: int = 5000,
    create_sample_if_missing: bool = True
):
    """Prepare warm-up data in the format expected by VERL SFT trainer."""

    data_path = Path(data_path)
    output_path = Path(output_path)

    # Check if warm-up data exists
    if not data_path.exists():
        if create_sample_if_missing:
            print(f"Warm-up data not found at {data_path}. Creating sample data...")
            create_sample_warmup_data(data_path)
        else:
            raise FileNotFoundError(f"Warm-up data not found at {data_path}")

    # Load and preprocess data
    df = pd.read_parquet(data_path)

    # Limit samples if specified
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} examples from {len(df)} total")

    # Ensure the data has the right columns for VERL SFT
    # VERL expects 'prompt' and 'response' columns (or configurable keys)
    if 'prompt' not in df.columns and 'question' in df.columns:
        df['prompt'] = df['question']

    if 'response' not in df.columns:
        if 'chosen' in df.columns:
            df['response'] = df['chosen']
        elif 'answer' in df.columns:
            df['response'] = df['answer']

    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[['prompt', 'response']].to_parquet(output_path, index=False)
    print(f"Saved {len(df)} warm-up samples to {output_path}")

    return output_path


def create_sample_warmup_data(output_path: Path):
    """Create sample warm-up data if none exists."""

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample data with proper format
    sample_data = []

    # Math reasoning examples
    math_prompts = [
        "Solve this math problem: What is 15% of 240?",
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "Calculate the area of a rectangle with length 8 cm and width 5 cm.",
        "What is the sum of all integers from 1 to 100?",
        "If you buy 3 apples for $1.50 each and 2 oranges for $0.75 each, what is the total cost?",
    ]

    math_responses = [
        "To find 15% of 240:\n15% = 0.15\n0.15 × 240 = 36\nTherefore, 15% of 240 is 36.",
        "To find average speed:\nDistance = 120 miles\nTime = 2 hours\nSpeed = Distance ÷ Time = 120 ÷ 2 = 60 mph\nThe average speed is 60 miles per hour.",
        "To find the area of a rectangle:\nArea = Length × Width\nArea = 8 cm × 5 cm = 40 cm²\nThe area is 40 square centimeters.",
        "The sum of integers from 1 to n is given by n(n+1)/2:\nSum = 100 × 101 ÷ 2 = 5050\nThe sum is 5050.",
        "Cost calculation:\n3 apples: 3 × $1.50 = $4.50\n2 oranges: 2 × $0.75 = $1.50\nTotal: $4.50 + $1.50 = $6.00",
    ]

    # Code generation examples
    code_prompts = [
        "Write a Python function to calculate factorial.",
        "Write a function to check if a number is prime.",
        "Write a function to reverse a string.",
        "Write a function to find the maximum element in a list.",
        "Write a function to check if a string is a palindrome.",
    ]

    code_responses = [
        "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n-1)",
        "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def reverse_string(s):\n    return s[::-1]",
        "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)",
        "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
    ]

    # General reasoning examples
    general_prompts = [
        "Explain the concept of recursion in programming.",
        "What is the difference between a list and a tuple in Python?",
        "Explain the time complexity of binary search.",
        "What is the purpose of inheritance in object-oriented programming?",
        "Describe the difference between deep learning and machine learning.",
    ]

    general_responses = [
        "Recursion is a programming technique where a function calls itself to solve a smaller instance of the same problem. It consists of a base case that stops the recursion and a recursive case that breaks down the problem.",
        "Lists are mutable (can be modified after creation) and use square brackets [], while tuples are immutable (cannot be modified) and use parentheses (). Lists are used for collections that may change, tuples for fixed collections.",
        "Binary search has O(log n) time complexity because it divides the search space in half with each comparison, reducing the problem size exponentially rather than linearly.",
        "Inheritance allows a class to inherit properties and methods from another class, promoting code reuse and establishing a hierarchy. It enables polymorphism and helps organize code in a logical structure.",
        "Machine learning uses algorithms to learn patterns from data, while deep learning is a subset that uses neural networks with multiple layers. Deep learning can automatically learn features, while traditional ML often requires manual feature engineering.",
    ]

    # Combine all examples
    for prompt, response in zip(math_prompts + code_prompts + general_prompts,
                                math_responses + code_responses + general_responses):
        sample_data.append({
            'prompt': prompt,
            'response': response,
        })

    # Duplicate to create more samples (for better training)
    sample_data = sample_data * 10  # Creates 150 samples

    # Save as parquet
    df = pd.DataFrame(sample_data)
    df.to_parquet(output_path, index=False)

    print(f"Created sample warm-up data with {len(df)} examples at {output_path}")


def create_warmup_config(
    model_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "checkpoints/warmup_sft",
    epochs: int = 2,
    batch_size: int = 256,
    micro_batch_size: int = 4,
    learning_rate: float = 5e-6,
    max_length: int = 2048,
    save_freq: int = 100,
    test_freq: int = 50,
    n_gpus: int = 8,
):
    """Create configuration for warm-up SFT training."""

    if val_data_path is None:
        val_data_path = train_data_path

    config = {
        'data': {
            'train_batch_size': batch_size,
            'micro_batch_size_per_gpu': micro_batch_size,
            'train_files': train_data_path,
            'val_files': val_data_path,
            'prompt_key': 'prompt',
            'response_key': 'response',
            'max_length': max_length,
            'truncation': 'error',
            'balance_dp_token': False,
            'chat_template': None,
            'custom_cls': {
                'path': None,
                'name': None,
            },
            'use_shm': False,
        },
        'model': {
            'partial_pretrain': model_path,
            'use_shm': False,
            'fsdp_config': {
                'model_dtype': 'bf16',
                'wrap_policy': {
                    'min_num_params': 0,
                },
                'cpu_offload': False,
                'offload_params': False,
            },
            'external_lib': None,
            'enable_gradient_checkpointing': True,
            'trust_remote_code': False,
            'lora_rank': 0,  # Set to 0 to disable LoRA, or use 32/64 for LoRA fine-tuning
            'lora_alpha': 16,
            'target_modules': 'all-linear',
            'use_liger': False,
            'strategy': 'fsdp2',
        },
        'optim': {
            'lr': learning_rate,
            'betas': [0.9, 0.95],
            'weight_decay': 0.01,
            'warmup_steps_ratio': 0.1,
            'clip_grad': 1.0,
            'lr_scheduler': 'cosine',
        },
        'ulysses_sequence_parallel_size': 1,
        'use_remove_padding': False,
        'trainer': {
            'default_local_dir': output_dir,
            'default_hdfs_dir': None,
            'resume_path': None,
            'project_name': 'warmup-sft',
            'experiment_name': 'warmup',
            'total_epochs': epochs,
            'total_training_steps': None,
            'logger': ['console'],  # Can add 'wandb' if needed
            'seed': 42,
            'save_freq': save_freq,
            'test_freq': test_freq,
            'nnodes': 1,
            'n_gpus_per_node': n_gpus,
            'max_ckpt_to_keep': 3,
        },
    }

    return DictConfig(config)


def main():
    """Main function to run warm-up SFT training."""

    import argparse
    parser = argparse.ArgumentParser(description="Warm-up SFT training using VERL")

    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the pretrained model')
    parser.add_argument('--data_path', type=str, default='data/warmup/curated_data.parquet',
                       help='Path to warm-up training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/warmup_sft',
                       help='Directory to save checkpoints')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Total batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4,
                       help='Micro batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of training samples')

    # Hardware configuration
    parser.add_argument('--n_gpus', type=int, default=2,
                       help='Number of GPUs to use')

    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Save checkpoint every N steps')
    parser.add_argument('--test_freq', type=int, default=50,
                       help='Validate every N steps')

    # LoRA configuration (optional)
    parser.add_argument('--lora_rank', type=int, default=0,
                       help='LoRA rank (0 to disable, 32/64 for LoRA)')

    args = parser.parse_args()

    # Prepare warm-up data
    processed_data_path = Path(args.output_dir) / "processed_warmup_data.parquet"
    processed_data_path = prepare_warmup_data(
        data_path=args.data_path,
        output_path=str(processed_data_path),
        max_samples=args.max_samples,
        create_sample_if_missing=True
    )

    # Create configuration
    config = create_warmup_config(
        model_path=args.model_path,
        train_data_path=str(processed_data_path),
        val_data_path=str(processed_data_path),  # Use same data for validation
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        n_gpus=args.n_gpus,
    )

    # Update LoRA configuration if specified
    if args.lora_rank > 0:
        config.model.lora_rank = args.lora_rank
        print(f"LoRA enabled with rank {args.lora_rank}")

    # Print configuration
    print("\n" + "="*50)
    print("Warm-up SFT Configuration")
    print("="*50)
    print(OmegaConf.to_yaml(config))
    print("="*50 + "\n")

    # Run SFT training
    print("Starting warm-up SFT training...")
    run_sft(config)

    print("\n" + "="*50)
    print("Warm-up SFT training completed!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()