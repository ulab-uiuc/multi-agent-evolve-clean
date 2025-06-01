abs_model_dir=$(realpath models)

bash eval_math_nodes.sh \
    --run_name azr_coder_3b_seed2 \
    --init_model $abs_model_dir/azr_coder-3b \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2

bash eval_math_nodes.sh \
    --run_name azr_base_7b_seed2 \
    --init_model $abs_model_dir/azr_base-7b \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2

bash eval_math_nodes.sh \
    --run_name azr_coder_7b_seed2 \
    --init_model $abs_model_dir/azr_coder-7b \
    --template azr_boxed \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2


bash eval_math_nodes.sh \
    --run_name azr_base_14b_seed2 \
    --init_model $abs_model_dir/azr_base-14b \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2


bash eval_math_nodes.sh \
    --run_name azr_coder_14b_seed2 \
    --init_model $abs_model_dir/azr_coder-14b \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,aime25,amc23,math500,olympiadbench,minerva_math \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2