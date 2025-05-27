# Evaluation of Absolute Zero Reasoner (AZR) on Math Benchmarks

### Requirements
You can install the required packages with the following command:
```bash
conda create -n azr_eval python==3.9
conda activate azr_eval
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
cd math_eval
tar -xzvf latex2sympy.tar.gz 
cd latex2sympy
pip install -e .
cd ../..
pip3 install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```


### Evaluation

First log into huggingface and download the model to be evaluated (or convterted AZR model trained by you): 
```bash
huggingface-cli download andrewzh/Absolute_Zero_Reasoner-Coder-3b --local-dir models/azr_3b --local-dir-use-symlinks False
```

Use the following script to evaluate AZR 3b on AIME24 benchmark with greedy decoding.

```bash
bash eval_math_nodes.sh \
    --run_name azr_3b_seed2 \
    --init_model <ABS_PATH_TO/eval/models/azr_3b> \
    --template azr \
    --tp_size 1 \
    --add_step_0 true \
    --temperature 0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24 \
    --n_sampling 1 \
    --just_wandb false \
    --seed 2
```


**Notes:**
- The `--init_model` must be the absolute path to the model directory.
- You should change `--template` if you are testing other models. It controls the prompt template used for the evaluation.
- Full list of benchmarks tested: `aime24,aime25,amc23,math500,olympiadbench,minerva_math,livemathbench`. See dataset under `data/` for other possible benchmarks.
   

## Acknowledgement
The codebase is adapted from [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason), which was based on [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
