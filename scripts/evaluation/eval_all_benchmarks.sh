#!/bin/bash
# Evaluate a MAE checkpoint on all general benchmarks (ID + OOD).
#
# Usage (auto-find step 150 under RUN_DIR):
#   RUN_DIR=./runs/MAE_3B_halfref_16 \
#   GLOBAL_STEP=150 \
#   bash scripts/evaluation/eval_all_benchmarks.sh
#
# Usage (explicit checkpoint path):
#   CKPT_PATH=/abs/path/to/global_step_150 \
#   RESUME_DIR=/abs/path/to/timestamped_run_dir \
#   bash scripts/evaluation/eval_all_benchmarks.sh
#
# Optional:
#   BENCHMARK_MAX_SAMPLES=500
#   CUDA_VISIBLE_DEVICES=0,1,2,3
set -x

export NCCL_DEBUG=INFO
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE=1

GLOBAL_STEP="${GLOBAL_STEP:-150}"
MODEL_TAG="${MODEL_TAG:-Qwen2.5-3B-Instruct}"
EXTRACTION_TYPE="${EXTRACTION_TYPE:-boxed}"
BENCHMARK_MAX_SAMPLES="${BENCHMARK_MAX_SAMPLES:-500}"

find_checkpoint() {
  local search_root="$1"
  local step="$2"
  find "${search_root}" -type d -path "*/general_io/${MODEL_TAG}/${EXTRACTION_TYPE}/global_step_${step}" 2>/dev/null | head -n 1
}

if [ -n "${CKPT_PATH:-}" ]; then
  CKPT_PATH="$(readlink -f "${CKPT_PATH}" 2>/dev/null || echo "${CKPT_PATH}")"
  if [ -z "${RESUME_DIR:-}" ]; then
    RESUME_DIR="$(dirname "$(dirname "$(dirname "${CKPT_PATH}")")")"
  fi
elif [ -n "${RUN_DIR:-}" ]; then
  CKPT_PATH="$(find_checkpoint "${RUN_DIR}" "${GLOBAL_STEP}")"
  if [ -z "${CKPT_PATH}" ]; then
    echo "Checkpoint global_step_${GLOBAL_STEP} not found under RUN_DIR=${RUN_DIR}"
    echo "Training checkpoints are saved under a timestamped subdir, e.g.:"
    echo "  ${RUN_DIR}/YYYYMMDD/HHMMSS_MAE_MAE_3B_halfref_16/general_io/${MODEL_TAG}/${EXTRACTION_TYPE}/global_step_${GLOBAL_STEP}"
    echo
    echo "Try searching from repo root:"
    echo "  find . -type d -name 'global_step_${GLOBAL_STEP}' 2>/dev/null"
    echo
    echo "Available checkpoints under ${RUN_DIR}:"
    find "${RUN_DIR}" -type d -name 'global_step_*' 2>/dev/null | sort || echo "  (none)"
    exit 1
  fi
  RESUME_DIR="$(dirname "$(dirname "$(dirname "${CKPT_PATH}")")")"
else
  echo "Set either CKPT_PATH or RUN_DIR."
  echo "Example:"
  echo "  RUN_DIR=./runs/MAE_3B_halfref_16 GLOBAL_STEP=150 bash scripts/evaluation/eval_all_benchmarks.sh"
  exit 1
fi

if [ ! -d "${CKPT_PATH}" ]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

echo "Using checkpoint: ${CKPT_PATH}"
echo "Using resume_dir: ${RESUME_DIR}"

# ID + OOD benchmarks used in this repo
ALL_BENCHMARKS="['mmlu', 'math', 'gsm8k', 'arc_challenge', 'gpqa', 'commonsenseqa', 'openbookqa', 'naturalquestions', 'triviaqa', 'squad', 'boolq', 'hellaswag', 'truthfulqa', 'bbh', 'livebench_reasoning', 'amc', 'minerva', 'winogrande', 'olympiad', 'mmlu_pro']"

python -m absolute_zero_reasoner.main_azr_ppo \
    --config-name=azr_ppo_trainer_general \
    +benchmark_names="${ALL_BENCHMARKS}" \
    +benchmark_max_samples="${BENCHMARK_MAX_SAMPLES}" \
    data.shuffle=True \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/code_reason/test_answer.parquet \
    data.val_files=data/code_reason/test_answer.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=512 \
    data.max_prompt_length=8192 \
    data.max_validation_prompt_length=6144 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.pretrained_tokenizer=True \
    +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.default_local_dir="${RESUME_DIR}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MAE' \
    trainer.experiment_name="MAE_3B_halfref_eval_step${GLOBAL_STEP}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.remove_previous_ckpt_in_save=False \
    trainer.del_local_ckpt_after_load=False \
    trainer.test_freq=50 \
    trainer.val_before_train=true \
    trainer.val_only=true \
    reward_fn.extraction_type="${EXTRACTION_TYPE}" \
    reward_fn.math_metric=deepscaler \
    reward_fn.llm_model_name="nvidia/llama-3.1-nemotron-70b-instruct" \
    reward_fn.temperature=1.0 \
    reward_fn.max_tokens=1000 \
    reward_fn.top_p=0.95 \
    reward_fn.stream=true \
    reward_fn.judge_with_actor=true \
    trainer.val_generations_to_log_to_wandb=0 \
    azr.task_type=general \
    azr.data_selection_strategy.update_iteration=1 \
    azr.pretrain_pred_steps=-1 \
    azr.problem_types=['general'] \
    azr.pred_data_mix_strategy=uniform_total \
    azr.judge_data_mix_strategy=uniform_total \
    azr.train_judge=True \
    azr.train_propose=True \
    azr.reward.n_samples=5 \
    azr.reward.generation_reward_config.format_reward=false \
    azr.reward.generation_reward_config.include_references=1 \
    azr.reward.generation_reward_config.generation_accuracy_convertion=inverse \
    azr.reward.generation_reward_config.answer_diversity_reward.hierarchical=false \
    azr.data_selection_strategy.content_max_length=8192 \
    azr.data_selection_strategy.max_questions=10000 \
    azr.data_selection_strategy.valid_question_filter=all \
    azr.data_selection_strategy.batched_estimate=false \
    azr.data_selection_strategy.io_n=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_dir="${RESUME_DIR}" \
    trainer.resume_from_path="${CKPT_PATH}" \
    trainer.total_epochs=1 \
    +prompt_manager.template_file=absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json \
    "$@"
