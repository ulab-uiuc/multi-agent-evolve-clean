#!/bin/bash

# Warm-up SFT training script using VERL's FSDP SFT Trainer
# This performs supervised fine-tuning on curated data before PPO training

# Configuration
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"  # Change to your model path
DATA_PATH="data/warmup/curated_data.parquet"  # Path to your warm-up data
OUTPUT_DIR="checkpoints/warmup_sft"
EPOCHS=5
BATCH_SIZE=256
MICRO_BATCH_SIZE=4
LEARNING_RATE=5e-6
MAX_LENGTH=2048
MAX_SAMPLES=5000
N_GPUS=2
SAVE_FREQ=5
TEST_FREQ=50
LORA_RANK=0  # Set to 0 to disable LoRA, or 32/64 to enable

# For distributed training
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --micro_batch_size)
      MICRO_BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --n_gpus)
      N_GPUS="$2"
      shift 2
      ;;
    --lora_rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p $OUTPUT_DIR

# Print configuration
echo "======================================"
echo "Warm-up SFT Training Configuration"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Micro Batch Size: $MICRO_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Length: $MAX_LENGTH"
echo "Max Samples: $MAX_SAMPLES"
echo "Number of GPUs: $N_GPUS"
echo "LoRA Rank: $LORA_RANK"
echo "======================================"

# Run the warm-up SFT training
if [ $N_GPUS -eq 1 ]; then
    # Single GPU training
    python scripts/warmup_sft.py \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --max_samples $MAX_SAMPLES \
        --n_gpus $N_GPUS \
        --save_freq $SAVE_FREQ \
        --test_freq $TEST_FREQ \
        --lora_rank $LORA_RANK
else
    # Multi-GPU training with torchrun
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$N_GPUS \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29500 \
        scripts/warmup_sft.py \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --max_samples $MAX_SAMPLES \
        --n_gpus $N_GPUS \
        --save_freq $SAVE_FREQ \
        --test_freq $TEST_FREQ \
        --lora_rank $LORA_RANK
fi

echo "======================================"
echo "Warm-up SFT training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "======================================"