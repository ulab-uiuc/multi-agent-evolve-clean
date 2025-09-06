import uuid
from typing import Optional
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Optional

from omegaconf import OmegaConf, open_dict
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, reduce_metrics, compute_data_metrics, compute_timing_metrics, AdvantageEstimator, compute_response_mask
from verl.utils.debug import marked_timer
_timer = marked_timer
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.trainer.ppo.ray_trainer import Role, WorkerType, ResourcePoolManager
from verl.utils.tracking import ValidationGenerationsLogger

from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset
from absolute_zero_reasoner.utils.benchmark_tracker import BenchmarkTracker


def compute_dr_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor):
    """
    Compute advantage for dr GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'dr_grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = compute_dr_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


class ReasonRLRayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        # Initialize benchmark evaluation for general tasks
        self.benchmark_reward_fn = None
        self.benchmark_dataloader = None
        if self._is_general_task():
            self._setup_benchmark_evaluation()
        
        # Initialize benchmark tracker for validation tracking
        self.benchmark_tracker = None
        if config.get('track_benchmarks', False):
            # Try to get output_dir from various config locations
            tracker_output_dir = None
            if hasattr(config, 'benchmark_tracker') and hasattr(config.benchmark_tracker, 'output_dir'):
                tracker_output_dir = config.benchmark_tracker.output_dir
            elif hasattr(config, 'trainer') and hasattr(config.trainer, 'output_dir'):
                tracker_output_dir = config.trainer.output_dir
            else:
                # Fallback to default directory
                tracker_output_dir = "./outputs/benchmark_tracking"
            
            print(f"[DEBUG] Initializing BenchmarkTracker with output_dir: {tracker_output_dir}")
            self.benchmark_tracker = BenchmarkTracker(
                output_dir=tracker_output_dir,
                config=config
            )
            print(f"[DEBUG] BenchmarkTracker initialized successfully")
        else:
            print(f"[DEBUG] BenchmarkTracker disabled (track_benchmarks={config.get('track_benchmarks', False)})")


    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _is_general_task(self) -> bool:
        """Check if the current task is a general task that should use benchmark evaluation."""
        # Try different ways to access the task type
        task_type = ""
        if hasattr(self.config, 'azr') and hasattr(self.config.azr, 'task_type'):
            task_type = self.config.azr.task_type
        else:
            task_type = self.config.get('azr', {}).get('task_type', '')
        
        print(f"[DEBUG] _is_general_task: task_type = '{task_type}'")
        return task_type.lower() == 'general'

    def _setup_benchmark_evaluation(self):
        """Setup benchmark evaluation for general tasks."""
        try:
            from absolute_zero_reasoner.rewards.reward_managers import BenchmarkEvaluationRewardManager
            from absolute_zero_reasoner.utils.benchmark_config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
            from torch.utils.data import DataLoader
            
            # Initialize benchmark config
            benchmark_config = BenchmarkConfig(
                validation_dir=self.config.get('benchmark_validation_dir', DEFAULT_BENCHMARK_CONFIG['validation_dir'])
            )
            
            # Get available benchmark files
            benchmark_names = self.config.get('benchmark_names', DEFAULT_BENCHMARK_CONFIG['default_benchmarks'])
            benchmark_files = benchmark_config.get_benchmark_files(benchmark_names)
            
            if not benchmark_files:
                print(f"Warning: No benchmark files found in {benchmark_config.validation_dir}")
                print(f"Expected benchmark names: {benchmark_names}")
                print("Skipping benchmark setup. To enable benchmark evaluation:")
                print("1. Prepare benchmark validation data in parquet format")
                print("2. Place files in the validation directory")
                print("3. Update benchmark_validation_dir in config if needed")
                self.benchmark_reward_fn = None
                self.benchmark_dataloader = None
                return
            
            print(f"Setting up benchmark evaluation with files: {benchmark_files}")
            
            # Create benchmark reward manager
            self.benchmark_reward_fn = BenchmarkEvaluationRewardManager(
                tokenizer=self.tokenizer,
                model_name=self.config.get('azr.benchmark_eval_model', "meta/llama-3.1-405b-instruct"),
                temperature=0.0,
                max_tokens=500
            )
            
            # Create benchmark dataset with per-benchmark sampling
            max_samples_per_benchmark = self.config.get('benchmark_max_samples', DEFAULT_BENCHMARK_CONFIG['max_samples_per_benchmark'])
            
            print(f"Loading benchmarks with max {max_samples_per_benchmark} samples per benchmark:")
            
            # Collect individual benchmark datasets with proper sampling
            benchmark_datasets = []
            total_samples = 0
            
            for benchmark_file in benchmark_files:
                try:
                    # Load single benchmark dataset
                    single_benchmark_dataset = RLHFDataset(
                        parquet_files=[benchmark_file],
                        tokenizer=self.tokenizer,
                        prompt_key=self.config.data.prompt_key,
                        max_prompt_length=self.config.data.get('max_validation_prompt_length', 8192),
                        filter_prompts=True,
                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                        truncation='error',
                        extra_source_key=f"benchmark_{benchmark_file.split('/')[-1].split('.')[0]}"
                    )
                    
                    benchmark_size = len(single_benchmark_dataset)
                    benchmark_name = benchmark_file.split('/')[-1].split('.')[0]
                    
                    # Apply per-benchmark sampling limit
                    if max_samples_per_benchmark and benchmark_size > max_samples_per_benchmark:
                        # Create subset with limited samples - Use fixed seed for reproducible question selection
                        generator = torch.Generator()
                        generator.manual_seed(42)  # Fixed seed ensures same questions every time
                        indices = torch.randperm(benchmark_size, generator=generator)[:max_samples_per_benchmark]
                        # Convert to python integers to avoid pandas indexing issues
                        indices = indices.tolist()
                        limited_dataset = torch.utils.data.Subset(single_benchmark_dataset, indices)
                        benchmark_datasets.append(limited_dataset)
                        actual_size = max_samples_per_benchmark
                        print(f"  {benchmark_name}: {actual_size}/{benchmark_size} samples (limited, fixed seed=42)")
                    else:
                        benchmark_datasets.append(single_benchmark_dataset)
                        actual_size = benchmark_size
                        print(f"  {benchmark_name}: {actual_size}/{benchmark_size} samples")
                    
                    total_samples += actual_size
                    
                except Exception as e:
                    print(f"Warning: Failed to load benchmark {benchmark_file}: {e}")
                    continue
            
            # Combine all benchmark datasets
            if benchmark_datasets:
                benchmark_dataset = torch.utils.data.ConcatDataset(benchmark_datasets)
                print(f"Total benchmark samples: {total_samples}")
                
                self.benchmark_dataloader = DataLoader(
                    dataset=benchmark_dataset,
                    batch_size=min(len(benchmark_dataset), 10),  # Use reasonable batch size
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn
                )
                
                print(f"Benchmark evaluation setup complete. Dataset size: {len(benchmark_dataset)}")
                print(f"[DEBUG] Fixed seed used for question selection to ensure consistency across evaluations")
            else:
                print("Warning: No benchmark datasets loaded successfully")
                self.benchmark_reward_fn = None
                self.benchmark_dataloader = None
            
        except Exception as e:
            print(f"Error setting up benchmark evaluation: {e}")
            print("Continuing without benchmark evaluation...")
            import traceback
            traceback.print_exc()
            self.benchmark_reward_fn = None
            self.benchmark_dataloader = None

    def _run_benchmark_evaluation(self) -> Dict:
        """Run benchmark evaluation and return metrics."""
        if self.benchmark_reward_fn is None or self.benchmark_dataloader is None:
            from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter as pp
            pp.status("BENCHMARK", "Benchmark evaluation not available (no data or reward function)", "warning")
            return {}
        
        from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter as pp
        
        pp.section_header("Running Benchmark Evaluation")
        print(f"[DEBUG] _run_benchmark_evaluation: Starting evaluation at step {self.global_steps}")
        
        reward_tensor_lst = []
        all_metrics = defaultdict(list)
        
        # Data for benchmark tracking
        all_questions = []
        all_model_answers = []
        all_ground_truths = []
        all_data_sources = []
        all_scores = []
        
        try:
            batch_count = 0
            for batch_data in self.benchmark_dataloader:
                batch_count += 1
                pp.status("BENCHMARK", f"Processing batch {batch_count}", "info")
                print(f"[DEBUG] _run_benchmark_evaluation: Processing batch {batch_count} with {len(batch_data.get('input_ids', []))} items")
                
                batch = DataProto.from_single_dict(batch_data)
                
                # Generate responses
                gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,  # Use greedy decoding for evaluation
                    'validate': True,
                }
                
                # Pad and generate
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
                
                # Combine batch with generated outputs
                batch = batch.union(output_gen_batch)
                
                # Store generated outputs for tracking
                output_ids = batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                
                # Collect benchmark tracking data if tracker is enabled
                if self.benchmark_tracker is not None:
                    print(f"[DEBUG] _run_benchmark_evaluation: Collecting tracking data from {len(batch)} items")
                    for i, data_item in enumerate(batch):
                        question = ""
                        ground_truth = ""
                        data_source = data_item.non_tensor_batch.get('data_source', 'benchmark_unknown')
                        
                        # Extract question and ground truth based on data structure
                        if 'prompt' in data_item.non_tensor_batch:
                            prompt_data = data_item.non_tensor_batch['prompt']
                            
                            # Handle numpy array case - convert to python list
                            if hasattr(prompt_data, 'tolist'):
                                prompt_data = prompt_data.tolist()
                            elif hasattr(prompt_data, 'item'):
                                prompt_data = prompt_data.item()
                            
                            if isinstance(prompt_data, list) and len(prompt_data) > 0:
                                first_prompt = prompt_data[0]
                                # Handle nested numpy array case
                                if hasattr(first_prompt, 'tolist'):
                                    first_prompt = first_prompt.tolist()
                                elif hasattr(first_prompt, 'item'):
                                    first_prompt = first_prompt.item()
                                
                                if isinstance(first_prompt, dict):
                                    question = first_prompt.get('content', '')
                                else:
                                    question = str(first_prompt) if first_prompt else ''
                        
                        if 'answer' in data_item.non_tensor_batch:
                            ground_truth = data_item.non_tensor_batch['answer']
                        elif 'reward_model' in data_item.non_tensor_batch:
                            ground_truth = data_item.non_tensor_batch['reward_model'].get('ground_truth', '')
                        
                        model_answer = output_texts[i] if i < len(output_texts) else ""
                        
                        all_questions.append(question)
                        all_model_answers.append(model_answer)
                        all_ground_truths.append(ground_truth)
                        all_data_sources.append(data_source)
                
                # Evaluate using benchmark reward manager
                reward_tensor, metrics = self.benchmark_reward_fn(batch)
                
                # Store scores for tracking
                scores = reward_tensor.sum(-1).cpu().tolist()
                if self.benchmark_tracker is not None:
                    all_scores.extend(scores)
                
                reward_tensor_lst.append(reward_tensor)
                for k, v in metrics.items():
                    all_metrics[k].append(v)
            
            if batch_count == 0:
                pp.status("BENCHMARK", "No batches found in benchmark dataloader", "warning")
                return {}
            
            # Record benchmark results if tracker is enabled
            if self.benchmark_tracker is not None and len(all_questions) > 0:
                print(f"[DEBUG] _run_benchmark_evaluation: Recording results for {len(all_questions)} items at step {self.global_steps}")
                
                # Group data by benchmark
                benchmark_data = defaultdict(list)
                for i in range(len(all_questions)):
                    benchmark_name = all_data_sources[i]
                    benchmark_data[benchmark_name].append({
                        'question': all_questions[i],
                        'model_answer': all_model_answers[i],
                        'ground_truth': all_ground_truths[i],
                        'score': all_scores[i],
                        'is_correct': all_scores[i] > 0.5  # Threshold for correctness
                    })
                
                # Record results for each benchmark
                for benchmark_name, data_list in benchmark_data.items():
                    questions = [item['question'] for item in data_list]
                    model_answers = [item['model_answer'] for item in data_list]
                    ground_truths = [item['ground_truth'] for item in data_list]
                    scores = [item['score'] for item in data_list]
                    is_correct = [item['is_correct'] for item in data_list]
                    
                    accuracy = np.mean(is_correct) if is_correct else 0.0
                    
                    self.benchmark_tracker.record_validation_results(
                        step=self.global_steps,
                        benchmark_name=benchmark_name,
                        questions=questions,
                        model_answers=model_answers,
                        ground_truths=ground_truths,
                        scores=scores,
                        accuracy=accuracy
                    )
            
            # Aggregate metrics
            final_metrics = {}
            for k, v_list in all_metrics.items():
                if v_list:
                    if isinstance(v_list[0], (int, float)):
                        final_metrics[k] = np.mean(v_list)
                    else:
                        final_metrics[k] = v_list[0]  # Take first value for non-numeric
            
            pp.status("Benchmark Evaluation", f"Completed successfully ({batch_count} batches processed)", "success")
            return final_metrics
            
        except Exception as e:
            pp.status("Benchmark Evaluation", f"Failed with error: {e}", "error")
            import traceback
            traceback.print_exc()
            return {}

    def _create_dataloader(self):
        """
        Changed the prompt length of validation set to have another prompt length.
        Create the train and val dataloader.
        """
        from torch.utils.data import RandomSampler, SequentialSampler
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         extra_source_key="train")
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       extra_source_key="val")
        self.val_dataloader = StatefulDataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    def _validate(self, do_sample: bool = False):
        """
        The validation loop of PPO.
        The only difference is logging more metrics.
        """
        from collections import defaultdict
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        all_eval_metrics = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': do_sample,
                'validate': True,
            }

            # pad to be divisible by dp_size
            size_divisor = self.actor_rollout_wg.world_size if not self.async_rollout_mode else self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]

            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, eval_metrics = self.val_reward_fn(test_batch)
            for k, v in eval_metrics.items():
                all_eval_metrics[k].append(v)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        for k, v in all_eval_metrics.items():
            metric_dict[k] = np.mean(v)

        if self.config.eval.get('save_generations', False):
            import json
            with open(f'{self.config.trainer.experiment_name}_generations_{self.global_steps}.json', 'w') as f:
                json.dump({
                    'inputs': sample_inputs,
                    'outputs': sample_outputs,
                    'scores': sample_scores
                }, f)
        return metric_dict


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.

        The only difference is logging more metrics.
        """
        from absolute_zero_reasoner.utils.tracking import ReasonRLTracking
        from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter as pp
        from omegaconf import OmegaConf

        # Display training setup header
        pp.section_header("Training Setup")

        logger = ReasonRLTracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            tags=self.config.trainer.wandb_tags,
            resume="must" if self.config.trainer.resume_mode == 'auto' and \
                self.config.trainer.wandb_run_id is not None else False,  # Add resume flag
            run_id=self.config.trainer.wandb_run_id \
                if self.config.trainer.wandb_run_id is not None else None  # Pass existing run ID
        )

        pp.status("Config", f"Project: {self.config.trainer.project_name}, Experiment: {self.config.trainer.experiment_name}", "info")
        pp.status("Algorithm", f"Using {self.config.algorithm.adv_estimator} advantage estimator", "info")
        pp.status("Setup", f"Critic enabled: {self.use_critic}, Reference policy: {self.use_reference_policy}", "info")

        self.global_steps = 0

        # load checkpoint before doing anything
        pp.status("Checkpoint", "Loading checkpoint if available...", "info")
        self._load_checkpoint()

        # base model chat template
        if self.config.actor_rollout_ref.model.pretrained_tokenizer:
            self.tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get('val_before_train', True) and self.global_steps == 0:
            val_metrics = {}
            if self._is_general_task():
                # For general tasks, only run benchmark evaluation
                if self.benchmark_reward_fn is not None:
                    pp.section_header("Initial Benchmark Evaluation")
                    pp.status("Benchmark", "Running initial benchmark evaluation...", "info")
                    val_metrics = self._run_benchmark_evaluation()
            else:
                # For other tasks, run standard validation
                if self.val_reward_fn is not None:
                    pp.section_header("Initial Validation")
                    pp.status("Validation", "Running initial validation...", "info")
                    val_metrics = self._validate(do_sample=self.config.eval.do_sample)

            if val_metrics:
                # Convert metrics to table format
                metrics_table = []
                for k, v in val_metrics.items():
                    metrics_table.append([k, f"{v:.4f}" if isinstance(v, float) else v])

                evaluation_type = "Initial Benchmark" if self._is_general_task() else "Initial Validation"
                pp.table(["Metric", "Value"], metrics_table, f"{evaluation_type} Results")
                logger.log(data=val_metrics, step=self.global_steps)

                # save val metrics to model path
                if self.config.eval.get('log_to_model_path', False):
                    import json
                    import os
                    with open(os.path.join(self.config.actor_rollout_ref.model.path, 'math_metrics.json'), 'w') as f:
                        json.dump(val_metrics, f)

            if self.config.trainer.get('val_only', False):
                pp.status("Training", "Validation only mode, exiting", "success")
                return

        # we start from step 1
        self.global_steps += 1
        total_steps = self.total_training_steps

        pp.section_header("Starting Training")
        pp.status("Training", f"Starting training for {self.config.trainer.total_epochs} epochs ({total_steps} steps)", "info")

        for epoch in range(self.config.trainer.total_epochs):
            pp.status("Epoch", f"Starting epoch {epoch+1}/{self.config.trainer.total_epochs}", "info")

            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        pp.status("Step", f"Generating sequences for batch {batch_idx+1}", "info")
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, _ = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    pp.status("Processing", "Preparing batch with UUIDs", "info")
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    pp.status("Processing", "Balancing batch across ranks", "info")
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = core_algos.agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer('values', timing_raw):
                            pp.status("Computation", "Computing critic values", "info")
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        pp.status("Rewards", "Computing rewards", "info")
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, train_metrics = self.reward_fn(batch)
                        train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
                        metrics.update(train_metrics)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            pp.status("KL Penalty", "Applying KL penalty", "info")
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        pp.status("Advantage", f"Computing {self.config.algorithm.adv_estimator} advantage", "info")
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with marked_timer('update_critic', timing_raw):
                            pp.status("Update", "Updating critic network", "info")
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer('update_actor', timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            pp.status("Update", "Updating actor network", "info")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=train_metrics,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer('testing', timing_raw):
                            is_general = self._is_general_task()
                            if is_general:
                                # For general tasks, only run benchmark evaluation
                                if self.benchmark_reward_fn is not None:
                                    pp.section_header(f"Benchmark Evaluation (Step {self.global_steps})")
                                    pp.status("Benchmark", "Running benchmark evaluation", "info")
                                    val_metrics: dict = self._run_benchmark_evaluation()
                                else:
                                    val_metrics = {}
                            else:
                                # For other tasks, run standard validation
                                if self.val_reward_fn is not None:
                                    pp.section_header(f"Validation (Step {self.global_steps})")
                                    pp.status("Validation", "Running validation", "info")
                                    val_metrics: dict = self._validate()
                                else:
                                    val_metrics = {}

                            # Convert metrics to table format
                            if val_metrics:
                                val_metrics_table = []
                                for k, v in val_metrics.items():
                                    val_metrics_table.append([k, f"{v:.4f}" if isinstance(v, float) else v])

                                evaluation_type = "Benchmark" if is_general else "Validation"
                                pp.table(["Metric", "Value"], val_metrics_table, f"{evaluation_type} Results (Step {self.global_steps})")
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer('save_checkpoint', timing_raw):
                            pp.status("Checkpoint", f"Saving checkpoint at step {self.global_steps}", "success")
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # Display key metrics in a table
                key_metrics = {k: v for k, v in metrics.items()}
                if key_metrics:
                    metrics_table = []
                    for k, v in key_metrics.items():
                        metrics_table.append([k, f"{v:.4f}" if isinstance(v, float) else v])
                    pp.table(["Metric", "Value"], metrics_table, f"Step {self.global_steps} Results")

                # Display timing info
                timing_metrics = {k: v for k, v in metrics.items() if 'time' in k}
                if timing_metrics:
                    timing_table = []
                    for k, v in timing_metrics.items():
                        timing_table.append([k, f"{v:.4f}s" if isinstance(v, float) else v])
                    pp.table(["Operation", "Time"], timing_table, "Timing Information")

                logger.log(data=metrics, step=self.global_steps)

                # Show progress within epoch
                pp.progress_bar(self.global_steps, total_steps, f"Training Progress (Epoch {epoch+1})")

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    pp.section_header("Training Complete")
                    # perform validation after training
                    val_metrics = {}
                    if self._is_general_task():
                        # For general tasks, only run benchmark evaluation
                        if self.benchmark_reward_fn is not None:
                            pp.status("Benchmark", "Running final benchmark evaluation", "info")
                            val_metrics = self._run_benchmark_evaluation()
                    else:
                        # For other tasks, run standard validation
                        if self.val_reward_fn is not None:
                            pp.status("Validation", "Running final validation", "info")
                            val_metrics = self._validate()

                    if val_metrics:
                        # Convert metrics to table format
                        final_metrics_table = []
                        for k, v in val_metrics.items():
                            final_metrics_table.append([k, f"{v:.4f}" if isinstance(v, float) else v])

                        evaluation_type = "Final Benchmark" if self._is_general_task() else "Final Validation"
                        pp.table(["Metric", "Value"], final_metrics_table, f"{evaluation_type} Results")
                        logger.log(data=val_metrics, step=self.global_steps)

                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with marked_timer('save_checkpoint', timing_raw):
                            pp.status("Checkpoint", "Saving final checkpoint", "success")
                            self._save_checkpoint()

                    pp.status("Training", "Training completed successfully!", "success")
                    return
