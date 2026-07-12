#!/bin/bash
# Half-ref MAE training on a 16-sample FusionBench seed subset (Qwen2.5-3B).
# Prerequisites:
#   1) python scripts/sample_fixed_fusionbench_subset.py --num_samples 16 --seed 42
#   2) python scripts/prepare_code_reason_placeholder.py --min-rows 16
#   3) api.json at repo root (NVIDIA NIM keys for judge reward)
#   4) Set RUN_DIR below (or pass trainer.default_local_dir=... on CLI)
set -x

export NCCL_DEBUG=INFO
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE=1

# Change this to your run output directory (or override via CLI).
RUN_DIR="${RUN_DIR:-./runs/MAE_3B_halfref_16}"

# Seed JSON lives at: ${DATA_DIR}/fixed_datasets/fixed_fusionbench_1000.json
# sample_fixed_fusionbench_subset.py default output uses data_16/ as default_data_dir.
DATA_DIR="${DATA_DIR:-./data_16}"

python -m absolute_zero_reasoner.main_azr_ppo \
    --config-name=azr_ppo_trainer_general \
    +benchmark_max_samples=100 \
    data.shuffle=True \
    actor_rollout_ref.ref.include_ref=False \
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
    trainer.default_local_dir="${RUN_DIR}" \
    trainer.default_data_dir="${DATA_DIR}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MAE' \
    trainer.experiment_name='MAE_3B_halfref_16' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.remove_previous_ckpt_in_save=False \
    trainer.del_local_ckpt_after_load=True \
    trainer.test_freq=50 \
    trainer.val_before_train=false \
    reward_fn.extraction_type=boxed \
    reward_fn.math_metric=deepscaler \
    reward_fn.llm_model_name="nvidia/llama-3.1-nemotron-70b-instruct" \
    reward_fn.temperature=1.0 \
    reward_fn.max_tokens=1000 \
    reward_fn.top_p=0.95 \
    reward_fn.stream=true \
    azr.task_type=general \
    azr.init_dataset_size=16 \
    azr.data_selection_strategy.update_iteration=1 \
    azr.pretrain_pred_steps=-1 \
    azr.problem_types=['general'] \
    azr.pred_data_mix_strategy=uniform_total \
    azr.judge_data_mix_strategy=uniform_total \
    azr.train_judge=True \
    azr.train_solve=True \
    azr.with_answer_generation=False \
    azr.train_propose=True \
    azr.reward.n_samples=5 \
    azr.reward.generation_reward_config.format_reward=false \
    azr.reward.generation_reward_config.include_references=0.5 \
    azr.reward.generation_reward_config.generation_accuracy_convertion=inverse \
    azr.reward.generation_reward_config.answer_diversity_reward.hierarchical=false \
    azr.data_selection_strategy.content_max_length=8192 \
    azr.data_selection_strategy.valid_question_filter=all \
    azr.data_selection_strategy.batched_estimate=false \
    azr.data_selection_strategy.io_n=1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=30 \
    +prompt_manager.template_file=absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json \
    "$@"
