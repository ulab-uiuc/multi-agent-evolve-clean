# Evaluation of Absolute Zero Reasoner (AZR) on Math Benchmarks

### Requirements
You can install the required packages with the following command:
```bash
conda create -n azr_eval python=3.10.16 -y
conda activate azr_eval

cd absolute_zero_reasoner/eval/math_eval
tar -xzvf latex2sympy.tar.gz 
cd latex2sympy
pip install -e .
cd ../..
pip install -r requirements.txt
```


### Evaluation

First log into huggingface and download the models to be evaluated (if you have not downloaded them yet):

```bash
# Download 3B Coder model
huggingface-cli download andrewzh/Absolute_Zero_Reasoner-Coder-3b --local-dir models/azr_coder-3b --local-dir-use-symlinks False

# Download 7B Coder model  
huggingface-cli download andrewzh/Absolute_Zero_Reasoner-Coder-7b --local-dir models/azr_coder-7b --local-dir-use-symlinks False

# Download 7B Base model
huggingface-cli download andrewzh2/Absolute_Zero_Reasoner-Base-7b --local-dir models/azr_base-7b --local-dir-use-symlinks False

# Download 14B Coder model
huggingface-cli download andrewzh/Absolute_Zero_Reasoner-Coder-14b --local-dir models/azr_coder-14b --local-dir-use-symlinks False

# Download 14B Base model
huggingface-cli download andrewzh2/Absolute_Zero_Reasoner-Base-14b --local-dir models/azr_base-14b --local-dir-use-symlinks False
```

Use the following script to evaluate AZR 7b on MATH500 benchmark with greedy decoding.

```bash
bash eval_math_nodes.sh \
    --run_name azr_base_7b_seed2 \
    --init_model $(realpath models/azr_base-7b) \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks math500 \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2
```


**Notes:**
- The `--init_model` must be the **absolute path** to your model directory. If you have downloaded them in a different directory, you should change it.
- You should change `--template` if you are testing other models. It controls the prompt template used for the evaluation.
- Full list of benchmarks tested: `aime24,aime25,amc23,math500,olympiadbench,minerva_math`. See dataset under `data/` for other possible benchmarks.
- You can change `--benchmarks` to test other benchmarks.
   

## Acknowledgement
The codebase is adapted from [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason), which was based on [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).