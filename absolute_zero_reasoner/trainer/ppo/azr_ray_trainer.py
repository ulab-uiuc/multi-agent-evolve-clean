import uuid
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Tuple
import random
import json
from collections import defaultdict
import threading
import gc
import os
import pickle
import ast

import ray
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from omegaconf import OmegaConf
import numpy as np
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.debug import marked_timer
_timer = marked_timer
from verl.trainer.ppo.ray_trainer import (
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    reduce_metrics,
    compute_timing_metrics,
    agg_loss,
)
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProto

from absolute_zero_reasoner.utils.tracking import ReasonRLTracking
from absolute_zero_reasoner.utils.prompt_manager import PromptManager
from absolute_zero_reasoner.utils.actor_prompt_optimizer import ActorPromptOptimizer, SafePromptUpdater
from absolute_zero_reasoner.data_construction.constructor import get_gen_code_io_data, get_pred_code_io_data, get_gen_general_io_data, get_pred_general_io_data, get_judge_general_io_data
from absolute_zero_reasoner.trainer.ppo.reason_rl_ray_trainer import ReasonRLRayPPOTrainer
from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset
from absolute_zero_reasoner.rewards.code_reward import parse_code_input_output, parse_inputs_message
from absolute_zero_reasoner.utils.code_utils.python_executor import PythonExecutor
from absolute_zero_reasoner.utils.code_utils.sandboxfusion_executor import SandboxfusionExecutor
from absolute_zero_reasoner.utils.auxiliary import reflection_keywords
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
from datasets import load_dataset
import random

seed_program = """def f(a):
    return a"""


def create_default_dict():
    return defaultdict(int)


def compute_data_metrics(batch, use_critic=True, tokenizer=None):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    reflect_list = []
    correct_list = []
    correct_response_length = []
    incorrect_response_length = []
    for i in range(len(batch)):
        data_item = batch[i]  # DataProtoItem
        prompt_ids = data_item.batch['prompts']
        _prompt_length = prompt_ids.shape[-1]
        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][_prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        # decode
        responses_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        reflect = any([kw in responses_str.lower() for kw in reflection_keywords])
        reflect_list.append(reflect)

        reward = data_item.batch['token_level_rewards'].sum(-1)
        correct = reward >= 1
        correct_list.append(correct)
        if correct:
            correct_response_length.append(valid_response_length.item())
        else:
            incorrect_response_length.append(valid_response_length.item())

    # the ratio of reflection
    reflect_ratio = sum(reflect_list) / len(reflect_list) \
        if len(reflect_list) > 0 else 0
    # the ratio of correct response in relfection samples
    correct_ratio = sum([reflect_list[i] and correct_list[i] for i in range(len(reflect_list))]) / \
        sum(reflect_list) if sum(reflect_list) > 0 else 0

    # separate lengths
    length_metrics = {}
    if len(correct_response_length) > 0:
        length_metrics['correct_response_length/mean'] = sum(correct_response_length) / len(correct_response_length)
    if len(incorrect_response_length) > 0:
        length_metrics['incorrect_response_length/mean'] = sum(incorrect_response_length) / len(incorrect_response_length)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        "response_length/reflect_ratio": reflect_ratio,
        "response_length/correct_reflect_ratio": correct_ratio,
        **length_metrics,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


# Create a local function to process elements before sending to manager
def process_elements(entries):
    """Process element types locally before sending to manager"""
    processed = []
    for entry in entries:
        entry_copy = entry.copy()
        if 'input' in entry:
            try:
                input_type = determine_type(entry['input'])
                entry_copy['_input_type'] = input_type
            except:
                entry_copy['_input_type'] = "str"

        if 'output' in entry:
            try:
                output_type = determine_type(entry['output'])
                entry_copy['_output_type'] = output_type
            except:
                entry_copy['_output_type'] = "str"
                
        if 'inputs' in entry:
            try:
                entry_copy['_input_types'] = [determine_type(inp) for inp in entry['inputs']]
            except:
                entry_copy['_input_types'] = ["str"] * len(entry['inputs'])
                
        if 'outputs' in entry:
            try:
                entry_copy['_output_types'] = [determine_type(out) for out in entry['outputs']]
            except:
                entry_copy['_output_types'] = ["str"] * len(entry['outputs'])

        processed.append(entry_copy)
    return processed

def determine_type(element):
    """Determine type safely without eval"""
    try:
        # Handle potential tuple strings without parentheses
        if isinstance(element, str) and ',' in element:
            # Attempt to parse as tuple by wrapping in parentheses
            try:
                wrapped = f'({element})'
                parsed_tuple = ast.literal_eval(wrapped)
                if isinstance(parsed_tuple, tuple):
                    return 'tuple'
            except:
                pass  # Proceed to normal parsing

        # Try using ast.literal_eval for safety
        parsed = ast.literal_eval(element)
        if is_pickleable(parsed):
            return type(parsed).__name__
        else:
            return "str"
    except:
        return "str"


def is_pickleable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


@ray.remote
class DatasetManager:
    def __init__(self):
        self.datasets = {
            'input': [],        # Stores only data entries
            'output': [],       # Stores only data entries 
            'seed': [],
            'error': [],
            'problem': [],
            'error_seed': [],
            'general': [],  # General dataset for mixed data
            'general_pair': [],
            'input_steps': [],  # Parallel list storing step numbers
            'output_steps': [], # Parallel list storing step numbers
            'error_steps': [], # Parallel list storing step numbers
            'problem_steps': [], # Parallel list storing step numbers
            'general_steps': [], # Parallel list storing step numbers
            'general_pair_steps': [],
            'input_steps_counter': defaultdict(int),
            'output_steps_counter': defaultdict(int),
            'error_steps_counter': defaultdict(int),
            'problem_steps_counter': defaultdict(int),
            'general_steps_counter': defaultdict(int),
            'general_pair_steps_counter': defaultdict(int),
        }
        self.type_counters = {
            'input_types': defaultdict(create_default_dict),
            'output_types': defaultdict(create_default_dict),
            'error_types': defaultdict(create_default_dict),
            'general_types': defaultdict(create_default_dict),
            'general_pair_types': defaultdict(create_default_dict),
        }
        self.locks = {
            'input': threading.Lock(),
            'output': threading.Lock(),
            'seed': threading.Lock(),
            'error': threading.Lock(),
            'problem': threading.Lock(),
            'general': threading.Lock(),
            'general_pair': threading.Lock(),
            'error_seed': threading.Lock(),
            'input_steps': threading.Lock(),
            'output_steps': threading.Lock(),
            'error_steps': threading.Lock(),
            'problem_steps': threading.Lock(),
            'general_steps': threading.Lock(),
            'general_pair_steps': threading.Lock(),
            'input_steps_counter': threading.Lock(),
            'output_steps_counter': threading.Lock(),
            'error_steps_counter': threading.Lock(),
            'problem_steps_counter': threading.Lock(),
            'general_steps_counter': threading.Lock(),
            'general_pair_steps_counter': threading.Lock(),
            'input_types': threading.RLock(),
            'output_types': threading.RLock(),
            'error_types': threading.RLock(),
            'general_types': threading.RLock(),
            'general_pair_types': threading.RLock(),
        }

    def update_seed(self, entries):
        with self.locks['seed']:
            existing = {json.dumps(d, sort_keys=True): True for d in self.datasets['seed']}
            new_entries = [e for e in entries if json.dumps(e, sort_keys=True) not in existing]

            for entry in new_entries:
                if 'input' in entry and '_input_type' in entry:
                    self.count_element(entry['input'], entry['_input_type'], 'input')
                if 'output' in entry and '_output_type' in entry:
                    self.count_element(entry['output'], entry['_output_type'], 'output')

            self.datasets['seed'].extend(new_entries)
            return len(new_entries)

    def update_error_seed(self, entries):
        with self.locks['error_seed'], self.locks['error_types']:
            existing = {json.dumps(d, sort_keys=True): True for d in self.datasets['error_seed']}
            new_entries = [e for e in entries if json.dumps(e, sort_keys=True) not in existing]

            # Process using pre-computed types
            for entry in new_entries:
                if 'output' in entry and '_output_type' in entry:
                    self.count_element(entry['output'], entry['_output_type'], 'error')

            self.datasets['error_seed'].extend(new_entries)
            return len(new_entries)

    def get_dataset(self, name) -> List[Dict]:
        """Returns only the data entries without step information"""
        return deepcopy(self.datasets[name])

    def get_all_datasets(self) -> Dict[str, List[Dict]]:
        """Returns all datasets without step information"""
        return {
            'input': deepcopy(self.datasets['input']),
            'output': deepcopy(self.datasets['output']),
            'seed': deepcopy(self.datasets['seed']),
            'error': deepcopy(self.datasets['error']),
            'problem': deepcopy(self.datasets['problem']),
            'general': deepcopy(self.datasets['general']),
            'general_pair': deepcopy(self.datasets['general_pair']),
            'error_seed': deepcopy(self.datasets['error_seed']),
            'input_steps': deepcopy(self.datasets['input_steps']),
            'output_steps': deepcopy(self.datasets['output_steps']),
            'error_steps': deepcopy(self.datasets['error_steps']),
            'problem_steps': deepcopy(self.datasets['problem_steps']),
            'general_steps': deepcopy(self.datasets['general_steps']),
            'general_pair_steps': deepcopy(self.datasets['general_pair_steps']),
            'input_steps_counter': deepcopy(self.datasets['input_steps_counter']),
            'output_steps_counter': deepcopy(self.datasets['output_steps_counter']),
            'error_steps_counter': deepcopy(self.datasets['error_steps_counter']),
            'problem_steps_counter': deepcopy(self.datasets['problem_steps_counter']),
            'general_steps_counter': deepcopy(self.datasets['general_steps_counter']),
        }

    def add_input_batch(self, entries: List[Dict], global_step: int):
        with self.locks['input'], self.locks['input_steps'], self.locks['input_types']:
            for entry in entries:
                if 'input' in entry and '_input_type' in entry:
                    self.count_element(entry['input'], entry['_input_type'], 'input')

            self.datasets['input'].extend(entries)
            self.datasets['input_steps'].extend([global_step]*len(entries))
            self.datasets['input_steps_counter'][global_step] += len(entries)
            return len(self.datasets['input'])

    def add_output_batch(self, entries: List[Dict], global_step: int):
        with self.locks['output'], self.locks['output_steps'], self.locks['output_types']:
            for entry in entries:
                if 'output' in entry and '_output_type' in entry:
                    self.count_element(entry['output'], entry['_output_type'], 'output')

            self.datasets['output'].extend(entries)
            self.datasets['output_steps'].extend([global_step]*len(entries))
            self.datasets['output_steps_counter'][global_step] += len(entries)
            return len(self.datasets['output'])

    def add_error_batch(self, entries: List[Dict], global_step: int):
        with self.locks['error'], self.locks['error_steps'], self.locks['error_types'], self.locks['error_types']:
            for entry in entries:
                if 'output' in entry and '_output_type' in entry:
                    self.count_element(entry['output'], entry['_output_type'], 'error')

            self.datasets['error'].extend(entries)
            self.datasets['error_steps'].extend([global_step]*len(entries))
            self.datasets['error_steps_counter'][global_step] += len(entries)
            return len(self.datasets['error'])

    def add_error_seed_batch(self, entries: List[Dict], global_step: int):
        with self.locks['error_seed'], self.locks['error_steps']:
            for entry in entries:
                if 'output' in entry and '_output_type' in entry:
                    self.count_element(entry['output'], entry['_output_type'], 'error')

            self.datasets['error_seed'].extend(entries)
            self.datasets['error_steps'].extend([global_step]*len(entries))
            self.datasets['error_steps_counter'][global_step] += len(entries)
            return len(self.datasets['error_seed'])

    def add_problem_batch(self, entries: List[Dict], global_step: int):
        with self.locks['problem'], self.locks['problem_steps'], self.locks['problem_steps_counter']:
            for entry in entries:
                if 'inputs' in entry and '_input_types' in entry:
                    for inp, inp_type in zip(entry['inputs'], entry['_input_types']):
                        self.count_element(inp, inp_type, 'input')
                if 'outputs' in entry and '_output_types' in entry:
                    for out, out_type in zip(entry['outputs'], entry['_output_types']):
                        self.count_element(out, out_type, 'output')

            self.datasets['problem'].extend(entries)
            self.datasets['problem_steps'].extend([global_step]*len(entries))
            self.datasets['problem_steps_counter'][global_step] += len(entries)
            return len(entries)
    
    def add_general_batch(self, entries: List[Dict], global_step: int):
        with self.locks['general'], self.locks['general_steps'], self.locks['general_steps_counter'], self.locks['general_types']:
            # Process type counting for general entries
            for entry in entries:
                # Count question types if available
                if 'question' in entry:
                    try:
                        question_type = determine_type(entry['question'])
                        self.count_element(entry['question'], question_type, 'general')
                    except:
                        pass
                # Count answer types if available
                if 'answer' in entry:
                    try:
                        answer_type = determine_type(entry['answer'])
                        self.count_element(entry['answer'], answer_type, 'general')
                    except:
                        pass
                # Count generation types if available
                if 'generation' in entry:
                    try:
                        generation_type = determine_type(entry['generation'])
                        self.count_element(entry['generation'], generation_type, 'general')
                    except:
                        pass

            self.datasets['general'].extend(entries)
            self.datasets['general_steps'].extend([global_step]*len(entries))
            self.datasets['general_steps_counter'][global_step] += len(entries)
            return len(entries)
        
    def add_general_pair_batch(self, entries: List[Dict], global_step: int):
        with self.locks['general_pair'], self.locks['general_pair_steps'], self.locks['general_pair_steps_counter'], self.locks['general_pair_types']:
            # Process type counting for general pair entries
            for entry in entries:
                if 'question' in entry:
                    try:
                        question_type = determine_type(entry['question'])
                        self.count_element(entry['question'], question_type, 'general_pair')
                    except:
                        pass
                if 'answer' in entry:
                    try:
                        answer_type = determine_type(entry['answer'])
                        self.count_element(entry['answer'], answer_type, 'general_pair')
                    except:
                        pass
                if 'generation' in entry:
                    try:
                        generation_type = determine_type(entry['generation'])
                        self.count_element(entry['generation'], generation_type, 'general_pair')
                    except:
                        pass

            self.datasets['general_pair'].extend(entries)
            self.datasets['general_pair_steps'].extend([global_step]*len(entries))
            self.datasets['general_pair_steps_counter'][global_step] += len(entries)
            return len(entries)

    def get_snippets(self) -> List[Dict]:
        # get the snippets from input and output datasets merged together
        snippets = []
        if self.datasets['input'] or self.datasets['output']:
            for d in self.datasets['input']:
                snippets.append({'snippet': d['snippet'], 'original_snippet': d['original_snippet'], 'imports': d['imports']})
            for d in self.datasets['output']:
                snippets.append({'snippet': d['snippet'], 'original_snippet': d['original_snippet'], 'imports': d['imports']})
            return list(snippets)
        else: # we are in the seed stage
            for d in self.datasets['seed']:
                snippets.append({'snippet': d['snippet'], 'original_snippet': d['original_snippet'], 'imports': d['imports']})
            return list(snippets)

    def get_snippets_with_steps(self) -> List[Tuple[Dict, int]]:
        snippets = self.get_snippets()
        return list(zip(snippets, self.datasets['input_steps'] + self.datasets['output_steps']))

    def get_recent_additions(self, dataset_key: str, current_step: int, window: int) -> int:
        counter_key = f"{dataset_key}_steps_counter"
        with self.locks[counter_key]:
            # Get steps from the counter dictionary instead of list
            recent_steps = [
                step for step in self.datasets[counter_key].keys()
                if (current_step - step) <= window
            ]
            total_recent = sum(
                self.datasets[counter_key][step] 
                for step in recent_steps
            )
            return total_recent

    def get_dataset_with_steps(self, name) -> List[Tuple[Dict, int]]:
        if name == 'input':
            assert len(self.datasets['input']) == len(self.datasets['input_steps']), \
                "Input data/steps mismatch!"
            return list(zip(deepcopy(self.datasets['input']), self.datasets['input_steps']))
        elif name == 'output':
            assert len(self.datasets['output']) == len(self.datasets['output_steps']), \
                "Output data/steps mismatch!"
            return list(zip(deepcopy(self.datasets['output']), self.datasets['output_steps']))
        elif name == 'error':
            assert len(self.datasets['error']) == len(self.datasets['error_steps']), \
                "Error data/steps mismatch!"
            return list(zip(deepcopy(self.datasets['error']), self.datasets['error_steps']))
        elif name == 'problem':
            assert len(self.datasets['problem']) == len(self.datasets['problem_steps']), \
                "Problem data/steps mismatch!"
            return list(zip(deepcopy(self.datasets['problem']), self.datasets['problem_steps']))
        elif name=="general":
            assert len(self.datasets['general']) == len(self.datasets['general_steps']), \
                "General data/steps mismatch!"
            return list(zip(deepcopy(self.datasets['general']), self.datasets['general_steps']))
        elif name == "general_pair":
            assert len(self.datasets['general_pair']) == len(self.datasets['general_pair_steps']), \
                "General pair data/steps mismatch!"
            return list(zip(deepcopy(self.datasets['general_pair']), self.datasets['general_pair_steps']))
        raise ValueError(f"Invalid dataset name: {name}")

    def get_steps_dataset(self, name) -> List[int]:
        if name == 'input':
            return self.datasets['input_steps']
        elif name == 'output':
            return self.datasets['output_steps']
        elif name == 'error':
            return self.datasets['error_steps']
        elif name == 'problem':
            return self.datasets['problem_steps']
        elif name == 'general':
            return self.datasets['general_steps']
        elif name == 'general_pair':
            return self.datasets['general_pair_steps']
        raise ValueError(f"Invalid dataset name: {name}")

    def truncate_datasets(self, max_length: int, name: str) -> Tuple[int, int]:
        if name == 'input':
            with self.locks['input'], self.locks['input_steps']:
                before_length = len(self.datasets['input'])
                self.datasets['input'] = self.datasets['input'][:max_length]
                self.datasets['input_steps'] = self.datasets['input_steps'][:max_length]
                truncated_length = before_length - len(self.datasets['input'])
                return truncated_length, before_length
        elif name == 'output':
            with self.locks['output'], self.locks['output_steps']:
                before_length = len(self.datasets['output'])
                self.datasets['output'] = self.datasets['output'][:max_length]
                self.datasets['output_steps'] = self.datasets['output_steps'][:max_length]
                truncated_length = before_length - len(self.datasets['output'])
                return truncated_length, before_length
        elif name == 'seed':
            with self.locks['seed']:
                before_length = len(self.datasets['seed'])
                self.datasets['seed'] = self.datasets['seed'][:max_length]
                truncated_length = before_length - len(self.datasets['seed'])
                return truncated_length, before_length
        elif name == 'error':
            with self.locks['error']:
                before_length = len(self.datasets['error'])
                self.datasets['error'] = self.datasets['error'][:max_length]
                truncated_length = before_length - len(self.datasets['error'])
                return truncated_length, before_length
        elif name == 'error_seed':
            with self.locks['error_seed']:
                before_length = len(self.datasets['error_seed'])
                self.datasets['error_seed'] = self.datasets['error_seed'][:max_length]
                truncated_length = before_length - len(self.datasets['error_seed'])
                return truncated_length, before_length
        elif name == 'problem':
            with self.locks['problem']:
                before_length = len(self.datasets['problem'])
                self.datasets['problem'] = self.datasets['problem'][:max_length]
                truncated_length = before_length - len(self.datasets['problem'])
                return truncated_length, before_length
        elif name == 'general':
            with self.locks['general']:
                before_length = len(self.datasets['general'])
                self.datasets['general'] = self.datasets['general'][:max_length]
                truncated_length = before_length - len(self.datasets['general'])
                return truncated_length, before_length
        elif name == 'general_pair':
            with self.locks['general_pair']:
                before_length = len(self.datasets['general_pair'])
                self.datasets['general_pair'] = self.datasets['general_pair'][:max_length]
                truncated_length = before_length - len(self.datasets['general_pair'])
                return truncated_length, before_length
        else:
            raise ValueError(f"Invalid dataset name: {name}")

    def get_dataset_size(self, name: str) -> int:
        with self.locks[name]:
            return len(self.datasets[name])

    def full_load_datasets(self, datasets):
        """Load all datasets from a dictionary"""
        self.datasets = datasets

    def full_load_data_with_type_counters(self, data: Dict):
        """Load datasets and type counters"""
        # First create a copy of the current empty structure
        default_structure = {
            'input': [], 'output': [], 'seed': [], 'error': [], 'problem': [],
            'error_seed': [], 'input_steps': [], 'output_steps': [], 'error_steps': [],
            'problem_steps': [], 'input_steps_counter': defaultdict(int),
            'general': [], 'general_pair': [], 'general_steps': [], 'general_pair_steps': [], 'general_steps_counter': defaultdict(int),
            'output_steps_counter': defaultdict(int), 'error_steps_counter': defaultdict(int), 'general_pair_steps_counter': defaultdict(int),
            'problem_steps_counter': defaultdict(int), 
        }
        
        # Extract datasets
        datasets_only = {k: v for k, v in data.items() if k != 'type_counters'}
        
        # Merge loaded data with default structure
        merged_datasets = default_structure.copy()
        merged_datasets.update(datasets_only)
        
        # Set the merged result
        self.datasets = merged_datasets

        # Then load type counters if available
        if 'type_counters' in data:
            with self.locks['input_types'], self.locks['output_types'], self.locks['error_types'], self.locks['general_types']:
                for counter_key in ['input_types', 'output_types', 'error_types', 'general_types']:
                    if counter_key in data['type_counters']:
                        self.type_counters[counter_key] = defaultdict(create_default_dict)
                        for type_name, values in data['type_counters'][counter_key].items():
                            for value, count in values.items():
                                self.type_counters[counter_key][type_name][value] = count

    def get_type_statistics(self, counter_key):
        """Get statistics about the types and their counts."""
        with self.locks[counter_key]:
            return {
                type_name: {
                    "total_unique": len(values),
                    "total_count": sum(values.values()),
                    "examples": list(values.keys())[:5]  # First 5 examples
                }
                for type_name, values in self.type_counters[counter_key].items()
            }

    def get_all_type_statistics(self):
        """Get all type statistics for inputs, outputs, and errors."""
        return {
            'input_types': self.get_type_statistics('input_types'),
            'output_types': self.get_type_statistics('output_types'),
            'error_types': self.get_type_statistics('error_types'),
            'general_types': self.get_type_statistics('general_types'),
            'general_pair_types': self.get_type_statistics('general_pair_types'),
        }

    def get_all_data_with_type_counters(self) -> Dict:
        """Returns all datasets and type counters"""
        all_data = self.get_all_datasets()
        all_data.update({
            'type_counters': {
                'input_types': deepcopy(self.type_counters['input_types']),
                'output_types': deepcopy(self.type_counters['output_types']),
                'error_types': deepcopy(self.type_counters['error_types']),
                'general_types': deepcopy(self.type_counters['general_types']),
                'general_pair_types': deepcopy(self.type_counters['general_pair_types']),
            }
        })
        return all_data

    def get_type_counter(self, counter_key):
        counter_type = f"{counter_key}_types"
        with self.locks[counter_type]:
            return self.type_counters[counter_type]

    def count_element(self, element, element_type, counter_key):
        counter_type = f"{counter_key}_types"
        with self.locks[counter_type]:
            self.type_counters[counter_type][element_type][element] += 1

class GeneralIORayPPOTrainer(ReasonRLRayPPOTrainer):

    def __init__(self, past_epoch_window: int = 10, benchmark_reward_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert self.config.actor_rollout_ref.rollout.n == 1, "GeneralIO only supports n=1 for now"

        self._past_epoch_window = past_epoch_window
        self.dataset_manager = DatasetManager.remote()
        self._last_cleanup_step = 0
        self._cleanup_frequency = self.config.azr.get('executor_cleanup_frequency', 5)
        self.benchmark_reward_fn = benchmark_reward_fn
        
        # Initialize prompt manager for dynamic prompt improvements
        self.prompt_manager = PromptManager(
            config=self.config,
            output_dir=self.config.trainer.default_local_dir + "/prompt_history"
        )
        
        # Initialize actor-driven prompt optimizer
        self.enable_actor_optimization = getattr(self.config.azr, 'enable_actor_prompt_optimization', False)
        if self.enable_actor_optimization:
            # Get infer_together configuration from reward_fn config
            infer_together = getattr(self.config.reward_fn, 'infer_together', False)
            
            self.actor_prompt_optimizer = ActorPromptOptimizer(
                model_interface=self._create_model_interface(),
                prompt_manager=self.prompt_manager,
                output_dir=self.config.trainer.default_local_dir + "/actor_prompt_optimization",
                infer_together=infer_together
            )
            self.safe_prompt_updater = SafePromptUpdater(self.prompt_manager)
            print(f"[DEBUG] GeneralIORayPPOTrainer: Initialized with actor prompt optimization ENABLED (infer_together={infer_together})")
        else:
            self.actor_prompt_optimizer = None
            self.safe_prompt_updater = None
            print(f"[DEBUG] GeneralIORayPPOTrainer: Actor prompt optimization DISABLED")
            
        # Set prompt_manager for reward functions if available
        if hasattr(self, 'prompt_manager') and self.prompt_manager is not None:
            if hasattr(self.reward_fn, 'set_prompt_manager'):
                self.reward_fn.set_prompt_manager(self.prompt_manager)
                print(f"[DEBUG] GeneralIORayPPOTrainer: Set prompt_manager for reward_fn")
            
            if hasattr(self, 'val_reward_fn') and self.val_reward_fn is not None and hasattr(self.val_reward_fn, 'set_prompt_manager'):
                self.val_reward_fn.set_prompt_manager(self.prompt_manager)
                print(f"[DEBUG] GeneralIORayPPOTrainer: Set prompt_manager for val_reward_fn")
                
            if hasattr(self, 'benchmark_reward_fn') and self.benchmark_reward_fn is not None and hasattr(self.benchmark_reward_fn, 'set_prompt_manager'):
                self.benchmark_reward_fn.set_prompt_manager(self.prompt_manager)
                print(f"[DEBUG] GeneralIORayPPOTrainer: Set prompt_manager for benchmark_reward_fn")
            
        print(f"[DEBUG] GeneralIORayPPOTrainer: Initialized with prompt manager")
    
    def _create_model_interface(self):
        """Create a model interface for prompt optimization"""
        class ActorModelInterface:
            def __init__(self, trainer):
                self.trainer = trainer
                
            def generate(self, prompt_text: str, max_length: int = 1024) -> str:
                """Generate text using the actor rollout worker group"""
                try:
                    # Tokenize input
                    tokenizer = self.trainer.tokenizer
                    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
                    
                    # Use actor rollout world size to ensure DataProto can be chunked evenly
                    world_size = getattr(self.trainer.actor_rollout_wg, 'world_size', 1)
                    
                    # Build a single-sample batch, we'll pad to divisor below
                    batch_dict = {
                        'input_ids': inputs['input_ids'],
                        'attention_mask': inputs['attention_mask'],
                        'position_ids': torch.arange(inputs['input_ids'].shape[1]).unsqueeze(0)
                    }
                    
                    # Convert to DataProto and pad to world_size divisor
                    gen_batch = DataProto.from_single_dict(batch_dict)
                    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
                    print(f"[DEBUG] ActorModelInterface: Padded batch for world_size={world_size}, pad_size={pad_size}")
                    
                    # Generate using actor_rollout_wg
                    with torch.no_grad():
                        gen_output_padded = self.trainer.actor_rollout_wg.generate_sequences(gen_batch_padded)
                    
                    # Unpad back to original size
                    gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)
                    
                    # Extract generated text (take the first result)
                    if 'responses' in gen_output.batch:
                        response_ids = gen_output.batch['responses'][0]
                        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                        return response_text.strip()
                    else:
                        print("[DEBUG] ActorModelInterface: No 'responses' in generation output")
                        return ""
                    
                except Exception as e:
                    print(f"[DEBUG] ActorModelInterface: Generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    return ""
        
        return ActorModelInterface(self)
    
    def cleanup(self):
        gc.collect()
    
    def _is_general_task(self) -> bool:
        """Check if current task type is general"""
        return getattr(self.config.azr, 'task_type', 'code') == 'general'

    def _run_benchmark_evaluation(self) -> dict:
        """Run benchmark evaluation for general tasks"""
        if self.benchmark_reward_fn is None:
            PrettyPrinter.status("BENCHMARK", "Benchmark reward function not available", "warn")
            return {}
        
        try:
            # Call the parent class method that's already implemented
            metrics = super()._run_benchmark_evaluation()
            PrettyPrinter.status("BENCHMARK", f"Completed benchmark evaluation with {len(metrics)} metrics", "success")
            if metrics:
                PrettyPrinter.status("BENCHMARK", f"Metrics keys: {list(metrics.keys())[:5]}", "info")
            return metrics
        except Exception as e:
            PrettyPrinter.status("BENCHMARK", f"Benchmark evaluation failed: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            return {}
    
    def _create_train_gen_dataloader(
        self,
        problem_type: str,
        data_len: int,
        dataset_key: str = None,
        seeding: bool = False,
    ) -> DataLoader:
        if dataset_key is None:
            dataset_key = "general"
        io_data = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))

        parquet_path = (self._code_dir / f'train_gen_{problem_type}.parquet').as_posix()
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        # Handle weights strategy
        if not seeding:
            weights = [1.0] * len(io_data)
        elif dataset_key == 'seed':
            weights = None
        else:
            weights = [1.0] * len(io_data)

        get_gen_general_io_data(
            io_data=io_data,
            io_n=self.config.azr.data_selection_strategy.io_n,
            target_data_len=data_len,
            content_max_length=self.config.azr.data_selection_strategy.content_max_length,
            output_path=parquet_path,
            split='train',
            tokenizer=self.tokenizer, 
            weights=weights,
            include_references=self.config.azr.reward.generation_reward_config.include_references,
            prompt_manager=self.prompt_manager,  # Pass prompt manager
        )

        gen_train_dataset = RLHFDataset(
            parquet_files=parquet_path,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            extra_source_key=f"gen_{problem_type}_train"
        )

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(gen_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(gen_train_dataset)

        return iter(DataLoader(
            dataset=gen_train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler
        ))
    
    def _create_train_pred_dataloader(self, problem_type: str, data_len: int) -> DataLoader:

        if problem_type not in {'general'}:
            raise ValueError(f"Invalid problem type for prediction: {problem_type}")
        dataset_key = "general"
        full_dataset = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))
        
        strategy = self.config.azr.pred_data_mix_strategy

        if strategy == "step":
            # Get entries with their creation steps
            entries_with_steps = ray.get(self.dataset_manager.get_dataset_with_steps.remote(dataset_key))
            if not entries_with_steps:
                selected_data = []
            else:
                entries, steps = zip(*entries_with_steps)
                # Calculate inverse step weights (newer entries get higher weight)
                selected_indices = random.choices(
                    range(len(entries)),
                    weights=steps,
                    k=min(data_len, len(entries))
                )
                selected_data = [entries[i] for i in selected_indices]
        elif strategy == "uniform_total":
            selected_data = random.sample(full_dataset, min(len(full_dataset), data_len))
        elif strategy == "max_new":
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_programs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_samples = random.sample(new_programs, min(len(new_programs), data_len))
            remaining = data_len - len(new_samples)
            selected_data = new_samples + random.sample(full_dataset, remaining)
        elif strategy == "half_new":
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_programs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_count = min(len(new_programs), data_len//2)
            base_count = data_len - new_count
            selected_data = random.sample(new_programs, new_count) + random.sample(full_dataset, base_count)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        parquet_path = (self._code_dir / f'train_pred_{problem_type}.parquet').as_posix()

        get_pred_general_io_data(
            io_data=selected_data,
            target_data_len=data_len,
            content_max_length=self.config.azr.data_selection_strategy.content_max_length,
            output_path=parquet_path,
            split='train',
            tokenizer=self.tokenizer,
            prompt_manager=self.prompt_manager,  # Pass prompt manager
        )

        general_pred_train_dataset = RLHFDataset(parquet_files=parquet_path,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         extra_source_key=f"pred_{problem_type}_train")
        with open('general_pred_train_dataset.txt', 'a') as f:
            f.write(f"Problem type: {problem_type}, Dataset size: {len(general_pred_train_dataset)}\n")
            f.write(f"Dataset: {general_pred_train_dataset}\n")
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=general_pred_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=general_pred_train_dataset)

        general_pred_train_dataloader = DataLoader(dataset=general_pred_train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        print(general_pred_train_dataloader)
        assert len(general_pred_train_dataloader) >= 1
        return iter(general_pred_train_dataloader)

    def _create_train_judge_dataloader(self, problem_type: str, data_len: int) -> DataLoader:
        if problem_type not in {'general'}:
            raise ValueError(f"Invalid problem type for judge: {problem_type}")
        dataset_key = "general_pair"
        # use pair data of question and answer for judging
        full_dataset = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))

        strategy = self.config.azr.judge_data_mix_strategy

        if strategy == "step":
            entries_with_steps = ray.get(self.dataset_manager.get_dataset_with_steps.remote(dataset_key))
            if not entries_with_steps:
                selected_data = []
            else:
                entries, steps = zip(*entries_with_steps)
                selected_indices = random.choices(
                    range(len(entries)),
                    weights=steps,
                    k=min(data_len, len(entries))
                )
                selected_data = [entries[i] for i in selected_indices]
        elif strategy == "uniform_total":
            selected_data = random.sample(full_dataset, min(len(full_dataset), data_len))
        elif strategy == "max_new":
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_programs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_samples = random.sample(new_programs, min(len(new_programs), data_len))
            remaining = data_len - len(new_samples)
            selected_data = new_samples + random.sample(full_dataset, remaining)
        elif strategy == "half_new":
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_programs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_count = min(len(new_programs), data_len//2)
            base_count = data_len - new_count
            selected_data = random.sample(new_programs, new_count) + random.sample(full_dataset, base_count)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        parquet_path = (self._code_dir / f'train_judge_{problem_type}.parquet').as_posix()

        get_judge_general_io_data(
            io_data=selected_data,
            target_data_len=data_len,
            content_max_length=self.config.azr.data_selection_strategy.content_max_length,
            output_path=parquet_path,
            split='train',
            tokenizer=self.tokenizer,
            prompt_manager=self.prompt_manager,  # Pass prompt manager
        )

        judge_train_dataset = RLHFDataset(
            parquet_files=parquet_path,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            extra_source_key=f"judge_{problem_type}_train"
        )
        with open('general_judge_train_dataset.txt', 'a') as f:
            f.write(f"Problem type: {problem_type}, Dataset size: {len(judge_train_dataset)}\n")
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=judge_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=judge_train_dataset)
        
        # Handle small datasets by adjusting batch_size and drop_last
        actual_batch_size = self.config.data.train_batch_size
        drop_last = True
        
        # If dataset is smaller than batch_size, adjust parameters
        if len(judge_train_dataset) < self.config.data.train_batch_size:
            if len(judge_train_dataset) == 0:
                PrettyPrinter.status("ERROR", f"Judge dataset is completely empty for {problem_type}. Judge training will be skipped.", "error")
                return None
            PrettyPrinter.status("WARN", f"Judge dataset too small to construct a complete batch. Judge training will be skipped", "error")
            return None
            # drop_last = False will cause problem in DataProto chunking
            # maybe fix later
            # else:
            #     # Use smaller batch size and don't drop last incomplete batch
            #     actual_batch_size = min(self.config.data.train_batch_size, len(judge_train_dataset))
            #     drop_last = False
            #     PrettyPrinter.status("WARN", f"Judge dataset size ({len(judge_train_dataset)}) is smaller than batch_size ({self.config.data.train_batch_size}). Using batch_size={actual_batch_size} and drop_last=False", "warn")
        
        judge_train_dataloader = DataLoader(dataset=judge_train_dataset,
                                           batch_size=actual_batch_size,
                                           drop_last=drop_last,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        print(judge_train_dataloader)
        
        # Verify dataloader has data when dataset is non-empty
        if len(judge_train_dataset) > 0:
            assert len(judge_train_dataloader) >= 1, f"Judge dataloader has length {len(judge_train_dataloader)} but dataset has {len(judge_train_dataset)} items (batch_size={actual_batch_size}, drop_last={drop_last})"
        else:
            PrettyPrinter.status("ERROR", f"Judge dataset is completely empty for {problem_type}. Judge training will be skipped.", "error")
            return None
        
        return iter(judge_train_dataloader)
    
    def _compute_batch(self, batch: DataProto, metrics: dict, timing_raw: dict, problem_type: str ) -> tuple[DataProto, dict]:
        PrettyPrinter.section_header(f"Computing batch for {problem_type}")
        # pop those keys for generation
        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

        # generate a batch
        with _timer(f'gen/{problem_type}', timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                dtype=object)
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        # balance the number of valid tokens on each dp rank
        self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

        # recompute old_log_probs
        with _timer(f'old_log_prob/{problem_type}', timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with _timer(f'ref/{problem_type}', timing_raw):
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with _timer(f'values/{problem_type}', timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with _timer(f'adv/{problem_type}', timing_raw):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)
            
            general_type_counters = ray.get(self.dataset_manager.get_type_statistics.remote('general_types'))
    
            # make sure actor_rollout_wg n > 1
            if problem_type.startswith('gen'):
                reward_fn_kwargs = {
                    'data': batch,
                    'problem_type': problem_type,
                    'rollout_actor_wg': self.actor_rollout_wg, # need this to estimate difficulty reward
                    'n_samples': self.config.azr.reward.n_samples,
                    'general_type_counters': general_type_counters,
                }
            elif problem_type.startswith('pred'):
                reward_fn_kwargs = {
                    'data': batch, 
                    'rollout_actor_wg': self.actor_rollout_wg,
                    'problem_type': problem_type,
                }
            elif problem_type.startswith('judge'):
                reward_fn_kwargs = {
                    'data': batch,
                    'rollout_actor_wg': self.actor_rollout_wg,
                    'problem_type': problem_type,
                }

            with _timer(f'reward_fn/{problem_type}', timing_raw):
                PrettyPrinter.status("REWARD", f"Computing rewards for {problem_type}...", "info")
                reward_tensor, train_metrics, valid_data = self.reward_fn(**reward_fn_kwargs)
                PrettyPrinter.status("REWARD", f"Found {len(valid_data) if valid_data else 0} valid data for {problem_type}", "success")

            # Log new questions, for now pairs will not be logged
            if valid_data and self.config.azr.random_print_max_programs > 0 and problem_type.startswith('gen'):
                PrettyPrinter.section_header(f"New {problem_type} Questions")
                max_print = min(self.config.azr.random_print_max_programs, len(valid_data))
                for question in random.sample(valid_data, max_print):
                    PrettyPrinter.status(f"PROBLEM TYPE", problem_type, "info")
                    PrettyPrinter.status("QUESTION", question['question'], "info")
                    PrettyPrinter.status("THOUGHT", question['thought'], "info")
                    PrettyPrinter.status("GENERATION", question['generation'], "info")
                    print("\n" + "-"*80 + "\n")
            
            if problem_type.startswith('gen'):
                ray.get(self.dataset_manager.add_general_batch.remote(valid_data, self.global_steps)) if valid_data else None
            if problem_type.startswith('pred'):
                ray.get(self.dataset_manager.add_general_pair_batch.remote(valid_data, self.global_steps)) if valid_data else None
                # valid_data are actually question-answer pairs here

            if self.config.azr.data_selection_strategy.max_questions is not None :
                truncated_length, before_length = ray.get(self.dataset_manager.truncate_datasets.remote(self.config.azr.data_selection_strategy.max_questions, 'general'))
                PrettyPrinter.status("DATA", f"Truncated {truncated_length} questions from general dataset, max questions is {self.config.azr.data_selection_strategy.max_questions}, dataset size was {before_length} before truncation", "info")
           
            train_metrics = {f'{problem_type}/{k}': np.mean(v) for k, v in train_metrics.items()}
            # log the number of valid programs added to the dataset
            if problem_type.startswith('gen'):
                if problem_type.endswith('general'):
                    dataset_key = 'general'
                else:
                    raise ValueError(f'Invalid problem type: {problem_type}')
                train_metrics[f'{problem_type}/num_valid_questions'] = ray.get(
                    self.dataset_manager.get_recent_additions.remote(
                        dataset_key, self.global_steps, self._past_epoch_window
                    )
                )
                # Dump questions to file
                if valid_data:
                    with open('question.txt', 'a') as f:
                        f.write(f"\n=== Global Training Step {self.global_steps} ===\n")
                        for item in valid_data:
                            f.write(f"Question: {item['question']}\n")
                            f.write("==============================================\n")
                            if 'thought' in item:
                                f.write(f"Thought: {item['thought']}\n")
                                f.write("==============================================\n")
                            if 'generation' in item:
                                f.write(f"Generation: {item['generation']}\n")
                                f.write("==============================================\n")
                            f.write("\n")
                        f.write("\n")
            if problem_type.startswith('pred'):
                if problem_type.endswith('general'):
                    dataset_key = 'general_pair'
                else:
                    raise ValueError(f'Invalid problem type: {problem_type}')
                train_metrics[f'{problem_type}/num_valid_pairs'] = ray.get(
                    self.dataset_manager.get_recent_additions.remote(
                        dataset_key, self.global_steps, self._past_epoch_window
                    )
                )
                # Dump question-answer pairs to file
                if valid_data:
                    with open('pair.txt', 'a') as f:
                        f.write(f"\n=== Global Training Step {self.global_steps} ===\n")
                        for item in valid_data:
                            if 'question' in item:
                                f.write(f"Question: {item['question']}\n")
                                f.write("==============================================\n")
                            if 'answer' in item:
                                f.write(f"Answer: {item['answer']}\n")
                                f.write("==============================================\n")
                            elif 'generation' in item:
                                f.write(f"Answer: {item['generation']}\n")
                                f.write("==============================================\n")
                            f.write("\n")
                        f.write("\n")
            metrics.update(train_metrics)
            batch.batch['token_level_scores'] = reward_tensor

            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                # Only apply KL penalty if use_kl_in_reward is enabled and kl_ctrl_in_reward is available
                if self.config.algorithm.use_kl_in_reward and hasattr(self, 'kl_ctrl_in_reward'):
                    batch, kl_metrics = apply_kl_penalty(batch,
                                                    kl_ctrl=self.kl_ctrl_in_reward,
                                                    kl_penalty=self.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)
                else:
                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
            else:
                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

            batch = compute_advantage(batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    config=self.config.algorithm)

        gc.collect()
        return batch, metrics

    def _init_seed_dataset(self):
        """Load fixed 1000 seed examples from FusionBench JSON file."""
        PrettyPrinter.section_header("Initializing GeneralIO Seed Dataset (from fixed data)")

        # Default data directory for storing fixed datasets
        base_data_dir = getattr(self.config.trainer, 'default_data_dir', None) or self.config.trainer.default_local_dir
        default_data_dir = Path(base_data_dir) / "fixed_datasets"
        default_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Primary path in the default data directory
        primary_data_path = default_data_dir / "fixed_fusionbench_1000.json"
        
        # Alternative paths to search for the fixed data
        alternative_paths = [
            Path(self.config.trainer.default_local_dir).parent / "data" / "fixed_fusionbench_1000.json",
            Path("data/fixed_fusionbench_1000.json"),
            Path("./fixed_fusionbench_1000.json"),
            Path(self.config.trainer.default_local_dir) / "fixed_fusionbench_1000.json",
        ]
        
        # Find the data file
        data_file_path = None
        if primary_data_path.exists():
            data_file_path = primary_data_path
        else:
            for alt_path in alternative_paths:
                if alt_path.exists():
                    data_file_path = alt_path
                    break
        
        if data_file_path is None:
            PrettyPrinter.status("ERROR", f"Fixed FusionBench data file not found. Searched paths:", "error")
            PrettyPrinter.status("ERROR", f"  - {primary_data_path} (default data dir)", "error")
            for alt_path in alternative_paths:
                PrettyPrinter.status("ERROR", f"  - {alt_path}", "error")
            PrettyPrinter.status("ERROR", "Please create the fixed data file using scripts/create_fixed_fusionbench_data.py", "error")
            PrettyPrinter.status("INFO", f"Default data directory created at: {default_data_dir}", "info")
            raise FileNotFoundError("Fixed FusionBench data file not found")
        
        try:
            # Load the fixed data
            with open(data_file_path, 'r', encoding='utf-8') as f:
                fixed_data = json.load(f)
            
            examples = fixed_data['examples']
            example_pairs = fixed_data['example_pairs']
            
            # If data was found in alternative location, copy it to default data dir
            if data_file_path != primary_data_path:
                PrettyPrinter.status("INFO", f"Copying data from {data_file_path} to default location {primary_data_path}", "info")
                with open(primary_data_path, 'w', encoding='utf-8') as f:
                    json.dump(fixed_data, f, ensure_ascii=False, indent=2)
            
            # Log metadata
            metadata = fixed_data.get('metadata', {})
            PrettyPrinter.status("INFO", f"Loaded fixed data from: {data_file_path}", "info")
            PrettyPrinter.status("INFO", f"Default data directory: {default_data_dir}", "info")
            PrettyPrinter.status("INFO", f"Data source: {metadata.get('source', 'Unknown')}", "info")
            PrettyPrinter.status("INFO", f"Seed: {metadata.get('seed', 'Unknown')}", "info")
            PrettyPrinter.status("INFO", f"Number of examples: {len(examples)}", "info")
            PrettyPrinter.status("INFO", f"Number of example pairs: {len(example_pairs)}", "info")
            
            # Use the fixed data directly (no shuffling since it's already fixed)
            seed_examples = examples[:1000]  # Take first 1000 or all if less
            seed_example_pairs = example_pairs[:1000]
            
        except Exception as e:
            PrettyPrinter.status("ERROR", f"Failed to load fixed FusionBench data: {str(e)}", "error")
            raise

        # Upload to ray
        ray.get(self.dataset_manager.add_general_batch.remote(seed_examples, self.global_steps))
        PrettyPrinter.status("SEED INIT", f"Successfully initialized with {len(seed_examples)} examples", "success")

        ray.get(self.dataset_manager.add_general_pair_batch.remote(seed_example_pairs, self.global_steps))
        PrettyPrinter.status("SEED INIT", f"Successfully initialized with {len(seed_example_pairs)} examples pairs", "success")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

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

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # base model chat template
        if self.config.actor_rollout_ref.model.pretrained_tokenizer:
            self.tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

        # currently, we only support validation using the reward_function.
        if self.config.trainer.get('val_before_train', True) and self.global_steps == 0:
            # PrettyPrinter.section_header(f"Starting Initial Validation")
            # val_metrics = self._validate()
            # PrettyPrinter.table(
            #     ["Metric", "Value"],
            #     [[k, v] for k, v in val_metrics.items()],
            #     title="Initial Validation Metrics"
            # )
            # logger.log(data=val_metrics, step=self.global_steps)
            # if self.config.trainer.get('val_only', False):
            #     return
            if self._is_general_task():
                PrettyPrinter.status("BENCHMARK", "Running benchmark evaluation for general task", "info")
                benchmark_metrics = self._run_benchmark_evaluation()
                
                # Debug: Print logger backends and metrics
                PrettyPrinter.status("DEBUG", f"Logger backends: {list(logger.logger.keys())}", "info")
                PrettyPrinter.status("DEBUG", f"Benchmark metrics type: {type(benchmark_metrics)}, length: {len(benchmark_metrics)}", "info")
                if benchmark_metrics:
                    PrettyPrinter.status("DEBUG", f"Sample metrics: {list(benchmark_metrics.keys())[:3]}", "info")
                
                if benchmark_metrics:
                    logger.log(data=benchmark_metrics, step=self.global_steps)
                    PrettyPrinter.status("WANDB", f"Logged {len(benchmark_metrics)} benchmark metrics to step {self.global_steps}", "success")
                else:
                    PrettyPrinter.status("WANDB", "No benchmark metrics to log", "warn")
                
                PrettyPrinter.table(
                    ["Benchmark Metric", "Value"],
                    [[k, v] for k, v in benchmark_metrics.items()],
                    title="Benchmark Evaluation Metrics"
                )
                
                if self.config.trainer.get('val_only', False):
                    return
        
        if self.config.trainer.val_only:
            PrettyPrinter.status("INFO", "Validation only mode enabled, exiting after validation", "info")
            return

        if self.loaded_datasets:
            PrettyPrinter.section_header(f"Resuming training from checkpoint")
            # print the lengths of the datasets
            for dataset_name in ['input', 'output', 'error', 'input_steps', 'output_steps', 'error_steps', 'input_steps_counter', 'output_steps_counter', 'error_steps_counter', 'general', 'general_steps', 'general_steps_counter']:
                PrettyPrinter.status("DATA", f"Length of {dataset_name}: {ray.get(self.dataset_manager.get_dataset_size.remote(dataset_name))}", "info")
        else:
            PrettyPrinter.section_header(f"Creating initial seed datasets")
            # create init dataset
            self._init_seed_dataset()

        # we start from step 1
        self.global_steps += 1
        if self.config.azr.pretrain_pred_steps > 0 and self.global_steps <= self.config.azr.pretrain_pred_steps:
            self.pretrain_pred = True
        else:
            self.pretrain_pred = False

        while self.global_steps < self.total_training_steps:
            PrettyPrinter.section_header(f"Training Step {self.global_steps}")
            #if self.config.azr.data_selection_strategy.composite_scheduler.enabled:
            #    self.scheduler_step()

            PrettyPrinter.progress_bar(
                current=self.global_steps,
                total=self.total_training_steps,
                label="Training Progress"
            )

            data_len = self.config.data.train_batch_size * self.config.azr.data_selection_strategy.update_iteration
            if "general" in self.config.azr.problem_types:
                gen_general_dataloader = self._create_train_gen_dataloader(
                    problem_type='general',
                    data_len=data_len,
                    dataset_key='general',
                )
                pred_general_dataloader = self._create_train_pred_dataloader(
                    problem_type='general',
                    data_len=data_len,
                )
                judge_general_dataloader = self._create_train_judge_dataloader(
                    problem_type='general',
                    data_len=data_len,
                )
                # judge general data only available for judging answer currently
                # more features like judging question may be available in the future
                # judge training is also only available now when training alone rather than together
                num_batches = len(gen_general_dataloader._loader) if hasattr(gen_general_dataloader, "_loader") else len(gen_general_dataloader)
                if num_batches < self.config.azr.data_selection_strategy.update_iteration:
                    PrettyPrinter.status("WARN", f"gen_general 只有 {num_batches} 批，不足 {self.config.azr.data_selection_strategy.update_iteration}；将重建/降级。", "warn")
            for _ in range(self.config.azr.data_selection_strategy.update_iteration):
                metrics = {}
                timing_raw = {}
                batches = {}
                with _timer('step', timing_raw):
                    # Clean up executor periodically
                    if self.global_steps - self._last_cleanup_step >= self._cleanup_frequency:
                        PrettyPrinter.section_header("Periodic Cleanup")
                        with _timer('cleanup', timing_raw):
                            self.cleanup()
                        self._last_cleanup_step = self.global_steps

                    if 'general' in self.config.azr.problem_types:
                        if not self.pretrain_pred:
                            try:
                                batch_dict = next(gen_general_dataloader)
                            except StopIteration:
                                gen_general_dataloader = self._create_train_gen_dataloader(
                                    problem_type='general',
                                    data_len=data_len,
                                    dataset_key='general',
                                )
                                batch_dict = next(gen_general_dataloader)
                            gen_batch: DataProto = DataProto.from_single_dict(batch_dict)
                            gen_batch, metrics = self._compute_batch(gen_batch, metrics, timing_raw, problem_type='gen_general')
                            if self.config.azr.train_propose:
                                batches[f'gen_general'] = gen_batch
                        batch_dict = next(pred_general_dataloader)
                        pred_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        pred_batch, metrics = self._compute_batch(pred_batch, metrics, timing_raw, problem_type='pred_general')
                        batches[f'pred_general'] = pred_batch
                        
                        # Handle judge training if dataloader is available and training is enabled
                        if judge_general_dataloader is not None and getattr(self.config.azr, 'train_judge', False):
                            try:
                                batch_dict = next(judge_general_dataloader)
                                judge_batch: DataProto = DataProto.from_single_dict(batch_dict)
                                judge_batch, metrics = self._compute_batch(judge_batch, metrics, timing_raw, problem_type='judge_general')
                                batches[f'judge_general'] = judge_batch
                            except StopIteration:
                                PrettyPrinter.status("WARN", "Judge dataloader exhausted, skipping judge training for this step", "warn")
                        elif judge_general_dataloader is None:
                            PrettyPrinter.status("INFO", "Judge training skipped due to insufficient paired data", "info")
                        else:
                            PrettyPrinter.status("INFO", "Judge training disabled in configuration", "info")

                    # concatenate batches
                    if not batches:
                        PrettyPrinter.status("ERROR", "No batches available for training. Skipping this step.", "error")
                        continue
                    
                    batch = DataProto.concat(list(batches.values()))

                    PrettyPrinter.section_header(f"Starting Parameter Updates")
                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    PrettyPrinter.section_header(f"Starting Validation")
                    if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            if self._is_general_task():
                                # For general tasks, only run benchmark evaluation
                                val_metrics: dict = self._run_benchmark_evaluation()
                                evaluation_type = "Benchmark"
                            else:
                                # For other tasks, run standard validation
                                if self.val_reward_fn is not None:
                                    val_metrics: dict = self._validate()
                                    evaluation_type = "Validation"
                                else:
                                    val_metrics = {}
                                    evaluation_type = "Validation"
                            
                            if val_metrics:
                                PrettyPrinter.table(
                                    ["Data Source", "Average Score"],
                                    [[k, v] for k, v in val_metrics.items()],
                                    title=f"{evaluation_type} Results"
                                )
                                
                                # Apply prompt improvements if benchmark tracker generated them
                                if hasattr(self, 'prompt_manager') and hasattr(self, 'benchmark_tracker'):
                                    try:
                                        # Get improvement prompts from benchmark tracker
                                        improvement_prompts = self.benchmark_tracker.generate_improvement_prompts(self.global_steps)
                                        
                                        if improvement_prompts:
                                            PrettyPrinter.section_header("Applying Prompt Improvements")
                                            
                                            # Get performance context
                                            analysis_report = self.benchmark_tracker.generate_analysis_report(self.global_steps)
                                            performance_context = f"Step {self.global_steps}: "
                                            if analysis_report and "Current overall accuracy:" in analysis_report:
                                                accuracy_lines = [line for line in analysis_report.split('\n') 
                                                                if 'Current overall accuracy:' in line]
                                                if accuracy_lines:
                                                    performance_context += accuracy_lines[0].strip()
                                            
                                            # Use actor-driven optimization if enabled
                                            if self.enable_actor_optimization and self.actor_prompt_optimizer:
                                                PrettyPrinter.status("ACTOR OPTIMIZATION", "Using actor model to optimize prompts", "info")
                                                
                                                # Get problematic questions and performance trends
                                                problematic_questions = self.benchmark_tracker.find_problematic_questions()
                                                performance_trends = self.benchmark_tracker.analyze_performance_trends()
                                                
                                                # Let actor model optimize prompts
                                                optimized_prompts = self.actor_prompt_optimizer.optimize_prompts_from_analysis(
                                                    benchmark_analysis=analysis_report,
                                                    problematic_questions=problematic_questions,
                                                    performance_trends=performance_trends,
                                                    step=self.global_steps
                                                )
                                                
                                                # Apply optimized prompts safely
                                                if optimized_prompts:
                                                    self.safe_prompt_updater.update_prompts_safely(
                                                        optimized_prompts=optimized_prompts,
                                                        step=self.global_steps,
                                                        performance_context=performance_context
                                                    )
                                                    
                                                    PrettyPrinter.status("ACTOR OPTIMIZATION", 
                                                                       f"Applied {len(optimized_prompts)} actor-optimized prompts", 
                                                                       "success")
                                                    
                                                    for prompt_type in optimized_prompts:
                                                        PrettyPrinter.status("  Actor Update", 
                                                                           f"{prompt_type}: Template updated by actor model", 
                                                                           "info")
                                                else:
                                                    PrettyPrinter.status("ACTOR OPTIMIZATION", 
                                                                       "No valid optimizations generated", 
                                                                       "warn")
                                            else:
                                                # Fallback to rule-based improvements
                                                PrettyPrinter.status("RULE-BASED", "Using rule-based prompt improvements", "info")
                                                
                                                # Apply improvements to prompt manager
                                                self.prompt_manager.update_prompts_from_analysis(
                                                    improvement_prompts=improvement_prompts,
                                                    performance_context=performance_context,
                                                    step=self.global_steps
                                                )
                                                
                                                # Log status
                                                prompt_status = self.prompt_manager.get_prompt_status()
                                                for prompt_type, status in prompt_status.items():
                                                    if status['improvements_count'] > 0:
                                                        PrettyPrinter.status("PROMPT UPDATE", 
                                                                           f"{prompt_type}: {status['improvements_count']} improvements", 
                                                                           "success")
                                                        for imp in status['current_improvements'][-3:]:  # Show last 3
                                                            PrettyPrinter.status("  Recent", imp[:100] + "...", "info")
                                        
                                    except Exception as e:
                                        PrettyPrinter.status("PROMPT UPDATE", f"Error applying improvements: {e}", "error")
                                        import traceback
                                        traceback.print_exc()
                        metrics.update(val_metrics)

                    # print the statistics of the number of questions in the dataset
                    if 'general' in self.config.azr.problem_types:
                        PrettyPrinter.status(
                            "DATA", 
                            f"Number of questions in the general dataset: {ray.get(self.dataset_manager.get_dataset_size.remote('general'))}", 
                            "info"
                        )
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics, separate problem types
                all_types = []
                if 'general' in self.config.azr.problem_types:
                    if not self.pretrain_pred:
                        all_types.append('gen_general')
                    all_types.append('pred_general')
                sep_batches = batch.chunk(len(all_types))
                for sep_batch, problem_type in zip(sep_batches, all_types):
                    sep_metrics = compute_data_metrics(batch=sep_batch, use_critic=self.use_critic, tokenizer=self.tokenizer)
                    sep_metrics = {f'{problem_type}/{k}': v for k, v in sep_metrics.items()}
                    metrics.update(sep_metrics)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # Get and log type statistics periodically
                type_stats = ray.get(self.dataset_manager.get_all_type_statistics.remote())

                # Log summary metrics about types
                for category, type_data in type_stats.items():
                    # Calculate diversity metrics
                    total_types = len(type_data)
                    total_unique_values = sum(stats["total_unique"] for stats in type_data.values())
                    total_instances = sum(stats["total_count"] for stats in type_data.values())

                    # Add to metrics
                    metrics[f"types/{category}/distinct_types"] = total_types
                    metrics[f"types/{category}/total_unique_values"] = total_unique_values
                    metrics[f"types/{category}/total_instances"] = total_instances

                    # Per-type metrics
                    for type_name, stats in type_data.items():
                        metrics[f"types/{category}/{type_name}/unique_count"] = stats["total_unique"]
                        metrics[f"types/{category}/{type_name}/total_count"] = stats["total_count"]

                # Print type statistics summary
                PrettyPrinter.section_header("Type Statistics Summary")
                for category, type_data in type_stats.items():
                    category_display = {
                        'input_types': 'Input Types',
                        'output_types': 'Output Types',
                        'error_types': 'Error Types',
                    }.get(category, category)

                    PrettyPrinter.status(category_display.upper(), f"Total types: {len(type_data)}", "info")
                    for type_name, stats in sorted(type_data.items(), 
                                                    key=lambda x: x[1]['total_unique'], 
                                                    reverse=True)[:5]:  # Show top 5 by unique count
                        PrettyPrinter.status(
                            f"  {type_name}", 
                            f"Unique: {stats['total_unique']}, Total: {stats['total_count']}", 
                            "info"
                        )
                        if stats['examples']:
                            example = stats['examples'][0]
                            if len(example) > 100:
                                example = example[:97] + "..."
                            PrettyPrinter.status("    Example", example, "info")

                PrettyPrinter.table(
                    ["Category", "Value"],
                    [[k, v] for k, v in metrics.items()],
                    title="Step Metrics"
                )

                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps >= self.config.azr.pretrain_pred_steps:
                    self.pretrain_pred = False

                self.global_steps += 1

                gc.collect()

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        PrettyPrinter.section_header(f"Starting Final Validation")
                        val_metrics = self._validate()
                        PrettyPrinter.table(
                            ["Data Source", "Average Score"],
                            [[k, v] for k, v in val_metrics.items()],
                            title="Final Validation Results"
                        )
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
                
    def _save_datasets(self, save_dir: Path):
        """Save input/output datasets as JSONL files"""
        save_dir.mkdir(parents=True, exist_ok=True)

        # get all datasets and type counters
        datasets_with_types = ray.get(self.dataset_manager.get_all_data_with_type_counters.remote())

        # save datasets
        pickle.dump(datasets_with_types, open(save_dir / 'datasets.pkl', 'wb'))
        PrettyPrinter.status("SAVE", f"Saved datasets and type counters to {save_dir}", "success")

    def _load_datasets(self, save_dir: Path):
        """Load input/output datasets from JSONL files"""
        datasets_with_types = pickle.load(open(Path(save_dir) / 'datasets' / 'datasets.pkl', 'rb'))

        # Filter datasets based on global step
        if self.global_steps > 0:
            # Filter datasets that have step info
            for dataset_key in ['input', 'output', 'error', 'problem', 'general']:
                steps_key = f"{dataset_key}_steps"
                if steps_key in datasets_with_types and dataset_key in datasets_with_types:
                    # Create lists of entries to keep
                    filtered_data = []
                    filtered_steps = []

                    # Only keep entries with steps less than current global_steps
                    for entry, step in zip(datasets_with_types[dataset_key], datasets_with_types[steps_key]):
                        if step <= self.global_steps:
                            filtered_data.append(entry)
                            filtered_steps.append(step)

                    # Update the datasets
                    datasets_with_types[dataset_key] = filtered_data
                    datasets_with_types[steps_key] = filtered_steps

                    # Also filter the step counter dictionaries
                    counter_key = f"{dataset_key}_steps_counter"
                    if counter_key in datasets_with_types:
                        filtered_counter = defaultdict(int)
                        for step, count in datasets_with_types[counter_key].items():
                            if step <= self.global_steps:
                                filtered_counter[step] = count
                        datasets_with_types[counter_key] = filtered_counter

            PrettyPrinter.status("FILTER", f"Filtered datasets to only include entries with steps <= {self.global_steps}", "info")

        ray.get(self.dataset_manager.full_load_data_with_type_counters.remote(datasets_with_types))
        PrettyPrinter.status("LOAD", f"Loaded datasets and type counters from {self.config.trainer.default_local_dir}", "success")
        self.loaded_datasets = True

    def _save_checkpoint(self):
        super()._save_checkpoint()
        # save datasets
        self._save_datasets(Path(self.config.trainer.default_local_dir) / 'datasets')
        PrettyPrinter.status("SAVE", f"Saved checkpoint to {self.config.trainer.default_local_dir}", "success")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps == 0:
            PrettyPrinter.section_header(f"Training from scratch")
        else:
            PrettyPrinter.section_header(f"Resuming training from checkpoint, step {self.global_steps}")

        # load datasets
        # first check if all the datasets exist
        code_dir = Path(self.config.trainer.default_local_dir) / 'code'
        self._code_dir = code_dir
        self.loaded_datasets = False
        if self.config.trainer.resume_mode == 'auto' and os.path.exists(os.path.join(self.config.trainer.default_local_dir, 'datasets', 'datasets.pkl')):
            self._load_datasets(self.config.trainer.default_local_dir)
        elif self.config.trainer.resume_mode == 'disable':
            if code_dir.exists():
                # delete all files and subdirectories in the code_dir
                for file in code_dir.glob('**/*'):
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        file.rmdir()
            PrettyPrinter.status("Directory", f"Cleaned existing code directory at {code_dir}", "info")
        elif not code_dir.exists():
            code_dir.mkdir(parents=True, exist_ok=True)
            PrettyPrinter.status("Directory", f"Created new code directory at {code_dir}", "info")

    def scheduler_step(self):
        if self.config.azr.data_selection_strategy.composite_scheduler.enabled:
            # Update number of programs - calculate directly based on global steps
            if self.global_steps >= self.config.azr.data_selection_strategy.composite_scheduler.update_num_programs_start:
                steps_since_start = self.global_steps - self.config.azr.data_selection_strategy.composite_scheduler.update_num_programs_start
                num_updates = steps_since_start // self.config.azr.data_selection_strategy.composite_scheduler.update_num_programs_interval

                # Calculate new value directly from initial value + increments
                initial_max = self.config.azr.data_selection_strategy.max_programs_initial
                new_max = min(initial_max + num_updates, self.config.azr.data_selection_strategy.composite_scheduler.num_programs_max)

                # Only log if value changed
                if new_max != self.config.azr.data_selection_strategy.composite_function_n_max:
                    current_max = self.config.azr.data_selection_strategy.composite_function_n_max
                    self.config.azr.data_selection_strategy.composite_function_n_max = new_max
                    PrettyPrinter.status("Scheduler", f"Updated max programs from {current_max} to {new_max}", "info")

            # Update composite probability - calculate directly based on global steps
            if self.global_steps >= self.config.azr.data_selection_strategy.composite_scheduler.update_probability_start:
                steps_since_start = self.global_steps - self.config.azr.data_selection_strategy.composite_scheduler.update_probability_start
                num_updates = steps_since_start // self.config.azr.data_selection_strategy.composite_scheduler.update_probability_interval

                # Calculate new value directly from initial value + increments
                initial_prob = self.config.azr.data_selection_strategy.composite_chance_initial
                new_prob = min(initial_prob + (num_updates * self.config.azr.data_selection_strategy.composite_scheduler.update_probability_increment),
                              self.config.azr.data_selection_strategy.composite_scheduler.update_probability_max)

                # Only log if value changed
                if new_prob != self.config.azr.data_selection_strategy.composite_chance:
                    current_prob = self.config.azr.data_selection_strategy.composite_chance
                    self.config.azr.data_selection_strategy.composite_chance = new_prob
                    PrettyPrinter.status("Scheduler", f"Updated composite probability from {current_prob:.2f} to {new_prob:.2f}", "info")



class CodeIORayPPOTrainer(ReasonRLRayPPOTrainer):
    _supported_tasks = {'code_i', 'code_o', 'code_e', 'code_f'}
    def __init__(self, past_epoch_window: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.config.actor_rollout_ref.rollout.n == 1, "CodeIO only supports n=1 for now"
        assert all(problem_type in self._supported_tasks for problem_type in self.config.azr.problem_types), \
            f"Invalid problem type: {self.config.azr.problem_types}"
        self._past_epoch_window = past_epoch_window
        if self.config.azr.executor == 'qwq':
            self._executor = PythonExecutor(
                timeout_length=self.config.azr.execute_max_timeout, 
                ast_check=self.config.azr.ast_check,
                max_workers=self.config.azr.get('executor_max_workers', 1)
            )
        else:
            raise ValueError(f'Invalid executor: {self.config.azr.executor}')
        self.dataset_manager = DatasetManager.remote()
        self._last_cleanup_step = 0
        self._cleanup_frequency = self.config.azr.get('executor_cleanup_frequency', 5)

    def cleanup(self):
        """Clean up the executor and other resources"""
        if hasattr(self._executor, 'cleanup'):
            PrettyPrinter.status("CLEANUP", "Cleaning up executor...", "info")
            self._executor.cleanup()
        # Force garbage collection
        gc.collect()

    def _create_train_code_gen_dataloader(
        self,
        problem_type: str,
        data_len: int,
        dataset_key: str = None,
        seeding: bool = False,
    ) -> DataLoader:
        if dataset_key is None:
            if problem_type == 'code_i':
                dataset_key = 'input'
            elif problem_type == 'code_o':
                dataset_key = 'output'
            elif problem_type == 'code_e':
                dataset_key = 'error'
            elif problem_type == 'code_f':
                # For code_f we use merged snippets from all datasets
                io_data = ray.get(self.dataset_manager.get_snippets.remote())
            else:
                raise ValueError(f'Invalid problem type: {problem_type}')
        
        if problem_type != 'code_f':
            io_data = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))

        parquet_path = (self._code_dir / f'train_gen_{problem_type}.parquet').as_posix()
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        # Handle weights strategy
        if problem_type == 'code_f' and not seeding:
            if self.config.azr.gen_data_probabilities_strategy == 'step':
                entries_with_steps = ray.get(self.dataset_manager.get_snippets_with_steps.remote())
                weights = [w + 1 for _, w in entries_with_steps] if entries_with_steps else [1.0]*len(io_data)
            else:
                weights = [1.0] * len(io_data)
        elif dataset_key == 'seed':
            weights = None
        elif self.config.azr.gen_data_probabilities_strategy == 'uniform':
            weights = [1.0] * len(io_data)
        elif self.config.azr.gen_data_probabilities_strategy == 'step':
            weights = [w + 1 for w in ray.get(self.dataset_manager.get_steps_dataset.remote(dataset_key))]
        else:
            raise ValueError(f"Unknown strategy: {self.config.azr.gen_data_probabilities_strategy}")

        # Common parameters for get_gen_code_io_data
        gen_params = {
            'io_data': io_data,
            'target_data_len': data_len,
            'problem_type': problem_type,
            'content_max_length': self.config.azr.data_selection_strategy.content_max_length,
            'io_n': 1 if problem_type == 'code_f' else self.config.azr.data_selection_strategy.io_n,
            'instruction_type': self.config.reward_fn.extraction_type,
            'output_path': parquet_path,
            'split': 'train',
            'tokenizer': self.tokenizer,
            'banned_keywords': self.config.azr.data_selection_strategy.banned_words,
            'banned_assertion_keywords': self.config.azr.data_selection_strategy.banned_keywords_for_errors_and_exceptions,
            'weights': weights,
            'enable_composite_function': self.config.azr.data_selection_strategy.composite_start_step > 0 and self.global_steps >= self.config.azr.data_selection_strategy.composite_start_step,
            'composite_function_n_min': self.config.azr.data_selection_strategy.composite_function_n_min,
            'composite_function_n_max': self.config.azr.data_selection_strategy.composite_function_n_max,
            'composite_chance': self.config.azr.data_selection_strategy.composite_chance,
            'remove_after_return': self.config.azr.reward.generation_reward_config.remove_after_return,
            'remove_input_from_snippet': self.config.azr.reward.generation_reward_config.remove_input_from_snippet,
            'include_references': self.config.azr.reward.generation_reward_config.include_references,
        }

        # Add code_f specific parameters
        if problem_type == 'code_f':
            gen_params.update({
                'num_inputs': self.config.azr.data_selection_strategy.num_inputs,
            })

        get_gen_code_io_data(**gen_params)

        code_gen_train_dataset = RLHFDataset(
            parquet_files=parquet_path,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            extra_source_key=f"gen_{problem_type}_train"
        )

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(code_gen_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(code_gen_train_dataset)

        return iter(DataLoader(
            dataset=code_gen_train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler
        ))

    def _create_train_code_pred_dataloader(self, problem_type: str, data_len: int) -> DataLoader:
        if problem_type == 'code_i':
            dataset_key = 'input'
        elif problem_type == 'code_o':
            dataset_key = 'output'
        elif problem_type == 'code_e':
            dataset_key = 'error'
        elif problem_type == 'code_f':
            dataset_key = 'problem'
        else:
            raise ValueError(f'Invalid problem type: {problem_type}')
        full_dataset = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))

        strategy = self.config.azr.pred_data_mix_strategy

        if strategy == "step":
            # Get entries with their creation steps
            entries_with_steps = ray.get(self.dataset_manager.get_dataset_with_steps.remote(dataset_key))
            if not entries_with_steps:
                selected_data = []
            else:
                entries, steps = zip(*entries_with_steps)
                # Calculate inverse step weights (newer entries get higher weight)
                selected_indices = random.choices(
                    range(len(entries)),
                    weights=steps,
                    k=min(data_len, len(entries))
                )
                selected_data = [entries[i] for i in selected_indices]
        elif strategy == "uniform_total":
            selected_data = random.sample(full_dataset, min(len(full_dataset), data_len))
        elif strategy == "max_new":
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_programs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_samples = random.sample(new_programs, min(len(new_programs), data_len))
            remaining = data_len - len(new_samples)
            selected_data = new_samples + random.sample(full_dataset, remaining)
        elif strategy == "half_new":
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_programs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_count = min(len(new_programs), data_len//2)
            base_count = data_len - new_count
            selected_data = random.sample(new_programs, new_count) + random.sample(full_dataset, base_count)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        parquet_path = (self._code_dir / f'train_pred_{problem_type}.parquet').as_posix()
        get_pred_code_io_data(
            io_data=selected_data,
            target_data_len=data_len,
            problem_type=problem_type,
            content_max_length=self.config.azr.data_selection_strategy.content_max_length,
            output_path=parquet_path,
            split='train',
            tokenizer=self.tokenizer,
            instruction_type=self.config.reward_fn.extraction_type,
        )
        code_pred_train_dataset = RLHFDataset(parquet_files=parquet_path,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         extra_source_key=f"pred_{problem_type}_train")
        with open('code_pred_train_dataset.txt', 'a') as f:
            f.write(f"Problem type: {problem_type}, Dataset size: {len(code_pred_train_dataset)}\n")
            f.write(f"Dataset: {code_pred_train_dataset}\n")
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=code_pred_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=code_pred_train_dataset)

        code_pred_train_dataloader = DataLoader(dataset=code_pred_train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        assert len(code_pred_train_dataloader) >= 1
        return iter(code_pred_train_dataloader)

    def _compute_batch(self, batch: DataProto, metrics: dict, timing_raw: dict, problem_type: str, executor: PythonExecutor) -> tuple[DataProto, dict]:
        PrettyPrinter.section_header(f"Computing batch for {problem_type}")
        # pop those keys for generation
        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

        # generate a batch
        with marked_timer(f'gen/{problem_type}', timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                dtype=object)
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        batch.batch["response_mask"] = compute_response_mask(batch)
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
        with marked_timer(f'old_log_prob/{problem_type}', timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
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
            with marked_timer(f'ref/{problem_type}', timing_raw):
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with marked_timer(f'values/{problem_type}', timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer(f'adv/{problem_type}', timing_raw):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            input_type_counters, output_type_counters, error_type_counters = None, None, None
            # Get the appropriate type counters based on problem type
            if problem_type == 'gen_code_i':
                input_type_counters = ray.get(self.dataset_manager.get_type_counter.remote('input'))
            elif problem_type == 'gen_code_o':
                output_type_counters = ray.get(self.dataset_manager.get_type_counter.remote('output'))
            elif problem_type == 'gen_code_e':
                error_type_counters = ray.get(self.dataset_manager.get_type_counter.remote('error'))
            elif problem_type == 'gen_code_f':
                input_type_counters = ray.get(self.dataset_manager.get_type_counter.remote('input'))
                output_type_counters = ray.get(self.dataset_manager.get_type_counter.remote('output'))

            # make sure actor_rollout_wg n > 1
            if problem_type.startswith('gen'):
                reward_fn_kwargs = {
                    'data': batch,
                    'problem_type': problem_type,
                    'executor': executor, # need this to check for execution errors
                    'rollout_actor_wg': self.actor_rollout_wg, # need this to estimate difficulty reward
                    'banned_words': self.config.azr.data_selection_strategy.banned_words, # need this to check for banned words
                    'n_samples': self.config.azr.reward.n_samples,
                    'input_type_counters': input_type_counters,
                    'output_type_counters': output_type_counters,
                    'error_type_counters': error_type_counters,
                }
            elif problem_type.startswith('pred'):
                reward_fn_kwargs = {
                    'data': batch, 
                    'problem_type': problem_type, 
                    'executor': executor,
                }
            with marked_timer(f'reward_fn/{problem_type}', timing_raw):
                PrettyPrinter.status("REWARD", f"Computing rewards for {problem_type}...", "info")
                reward_tensor, train_metrics, valid_programs, correct_predictions = self.reward_fn(**reward_fn_kwargs)
                PrettyPrinter.status("REWARD", f"Found {len(valid_programs) if valid_programs else 0} valid programs", "success")

                # get avg_program lines
                avg_program_lines = sum(len(program['snippet'].split('\n')) for program in valid_programs) / len(valid_programs) if valid_programs else 0
                train_metrics[f'{problem_type}/avg_program_lines'] = avg_program_lines

            # Log new programs if available
            if valid_programs and self.config.azr.random_print_max_programs > 0:
                PrettyPrinter.section_header(f"New {problem_type} Programs")
                max_print = min(self.config.azr.random_print_max_programs, len(valid_programs))
                for program in random.sample(valid_programs, max_print):
                    PrettyPrinter.status(f"PROBLEM TYPE", problem_type, "info")
                    if 'code_f' not in problem_type:
                        PrettyPrinter.code_block(program['snippet'], "python")
                        PrettyPrinter.status("INPUT", program['input'], "info")
                        PrettyPrinter.status("OUTPUT", program['output'], "info")
                        PrettyPrinter.status("THOUGHT", program['thought'], "info")
                        PrettyPrinter.status("COMPOSITE FUNCTION", "YES!" if len(program['composite_functions']) > 0 else "NO!", "info")
                    else:
                        PrettyPrinter.code_block(program['snippet'], "python")
                        PrettyPrinter.status("INPUT", program['inputs'], "info")
                        PrettyPrinter.status("OUTPUT", program['outputs'], "info")
                        PrettyPrinter.status("MESSAGE", program['message'], "info")
                        PrettyPrinter.status("THOUGHT", program['thought'], "info")
                    print("\n" + "-"*80 + "\n")
            if correct_predictions and self.config.azr.random_print_max_programs > 0:
                PrettyPrinter.section_header(f"New {problem_type} Programs")
                max_print = min(self.config.azr.random_print_max_programs, len(correct_predictions))
                for program in random.sample(correct_predictions, max_print):
                    if 'code_f' not in problem_type:
                        PrettyPrinter.code_block(program['program'], "python")
                        # also print the problem_type
                        PrettyPrinter.status(f"PROBLEM TYPE", problem_type, "info")
                        PrettyPrinter.status("INPUT", program['input'], "info")
                        PrettyPrinter.status("OUTPUT", program['output'], "info")
                        PrettyPrinter.status("THOUGHT", program['thought'], "info")
                    else:
                        PrettyPrinter.code_block(program['answer']['snippet'], "python")
                        PrettyPrinter.code_block(program['answer']['gold_program'], "python (gold)")
                        PrettyPrinter.status("HIDDEN INPUT", program['hidden_inputs'], "info")
                        PrettyPrinter.status("HIDDEN OUTPUT", program['hidden_outputs'], "info")
                        PrettyPrinter.status("GIVEN INPUT", program['given_inputs'], "info")
                        PrettyPrinter.status("GIVEN OUTPUT", program['given_outputs'], "info")
                        PrettyPrinter.status("MESSAGE", program['answer']['message'], "info")
                        PrettyPrinter.status("THOUGHT", program['answer']['thought'], "info")
                    print("\n" + "-"*80 + "\n")

            if problem_type.endswith('code_i'):
                if valid_programs:
                    # Process locally first
                    processed_programs = process_elements(valid_programs)
                    # Then batch add to dataset
                    ray.get(self.dataset_manager.add_input_batch.remote(processed_programs, self.global_steps))
            elif problem_type.endswith('code_o'):
                if valid_programs:
                    processed_programs = process_elements(valid_programs)
                    ray.get(self.dataset_manager.add_output_batch.remote(processed_programs, self.global_steps))
            elif problem_type.endswith('code_e'):
                if valid_programs:
                    processed_programs = process_elements(valid_programs)
                    ray.get(self.dataset_manager.add_error_batch.remote(processed_programs, self.global_steps))
            elif problem_type.endswith('code_f'):
                if valid_programs:
                    processed_programs = process_elements(valid_programs)
                    ray.get(self.dataset_manager.add_problem_batch.remote(processed_programs, self.global_steps))
            else:
                raise ValueError(f'Invalid problem type: {problem_type}')

            if self.config.azr.data_selection_strategy.max_programs is not None and problem_type.endswith('code_i'):
                truncated_length, before_length = ray.get(self.dataset_manager.truncate_datasets.remote(self.config.azr.data_selection_strategy.max_programs, 'input'))
                PrettyPrinter.status("DATA", f"Truncated {truncated_length} programs from input dataset, max programs is {self.config.azr.data_selection_strategy.max_programs}, dataset size was {before_length} before truncation", "info")
            if self.config.azr.data_selection_strategy.max_programs is not None and problem_type.endswith('code_o'):
                truncated_length, before_length = ray.get(self.dataset_manager.truncate_datasets.remote(self.config.azr.data_selection_strategy.max_programs, 'output'))
                PrettyPrinter.status("DATA", f"Truncated {truncated_length} programs from output dataset, max programs is {self.config.azr.data_selection_strategy.max_programs}, dataset size was {before_length} before truncation", "info")
            if self.config.azr.data_selection_strategy.max_programs is not None and problem_type.endswith('code_e'):
                truncated_length, before_length = ray.get(self.dataset_manager.truncate_datasets.remote(self.config.azr.data_selection_strategy.max_programs, 'error'))
                PrettyPrinter.status("DATA", f"Truncated {truncated_length} programs from error dataset, max programs is {self.config.azr.data_selection_strategy.max_programs}, dataset size was {before_length} before truncation", "info")
            if self.config.azr.data_selection_strategy.max_programs is not None and problem_type.endswith('code_f'):
                truncated_length, before_length = ray.get(self.dataset_manager.truncate_datasets.remote(self.config.azr.data_selection_strategy.max_programs, 'problem'))
                PrettyPrinter.status("DATA", f"Truncated {truncated_length} programs from problem dataset, max programs is {self.config.azr.data_selection_strategy.max_programs}, dataset size was {before_length} before truncation", "info")

            train_metrics = {f'{problem_type}/{k}': np.mean(v) for k, v in train_metrics.items()}
            # log the number of valid programs added to the dataset
            if problem_type.startswith('gen'):
                if problem_type.endswith('code_i'):
                    dataset_key = 'input' 
                elif problem_type.endswith('code_o'):
                    dataset_key = 'output'
                elif problem_type.endswith('code_e'):
                    dataset_key = 'error'
                elif problem_type.endswith('code_f'):
                    dataset_key = 'problem'
                else:
                    raise ValueError(f'Invalid problem type: {problem_type}')
                train_metrics[f'{problem_type}/num_valid_programs'] = ray.get(
                    self.dataset_manager.get_recent_additions.remote(
                        dataset_key, self.global_steps, self._past_epoch_window
                    )
                )
            metrics.update(train_metrics)
            batch.batch['token_level_scores'] = reward_tensor

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch,
                                                kl_ctrl=self.kl_ctrl_in_reward,
                                                kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

            batch = compute_advantage(batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    config=self.config.algorithm)

        gc.collect()
        return batch, metrics

    def _init_seed_dataset(self, problem_types: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        # Initialize with seed program using the coordinator
        if ('code_i' in problem_types or 'code_o' in problem_types) and ray.get(self.dataset_manager.get_dataset.remote('seed')) == []:
            ray.get(self.dataset_manager.update_seed.remote([
                {'snippet': seed_program, 'input': '"Hello world"', 'output': '"Hello world"', 'imports': [], 'original_snippet': seed_program, 'composite_functions': []}
            ]))
        if 'code_e' in problem_types and ray.get(self.dataset_manager.get_dataset.remote('error_seed')) == []:
            ray.get(self.dataset_manager.update_error_seed.remote([
                {'snippet': seed_program, 'input': '"Hello world"', 'output': 'NoError', 'imports': [], 'original_snippet': seed_program, 'composite_functions': []}
            ]))
        if 'code_f' in problem_types and ray.get(self.dataset_manager.get_dataset.remote('problem')) == []:
            ray.get(self.dataset_manager.add_problem_batch.remote([
                {
                    'snippet': seed_program,
                    'inputs': ['"Hello world"', '1', "dict(a=1, b=2)", '(1.1, 1.2, 1.3)', '"[[1, 0, 0], [0, 0, 0], [0, 0, 0]]"', '1001101100010001'],
                    'outputs': ['"Hello world"', '1', "dict(a=1, b=2)", '(1.1, 1.2, 1.3)', '"[[1, 0, 0], [0, 0, 0], [0, 0, 0]]"', '1001101100010001'],
                    'message': 'Write a function that returns whatever you input',
                    'imports': [],
                }
            ], self.global_steps))

        target_size = self.config.azr.data_selection_strategy.data_len * self.config.azr.data_selection_strategy.seed_batch_factor

        while problem_types != ['code_f']: # we can skip this loop if we are only generating code_f dataset
            # Get current dataset state
            seed_dataset = ray.get(self.dataset_manager.get_dataset.remote('seed'))
            error_dataset = ray.get(self.dataset_manager.get_dataset.remote('error_seed'))
            if problem_types == ['code_e'] and len(error_dataset) >= target_size: # only generate error seed dataset
                break
            if 'code_e' not in problem_types and len(seed_dataset) >= target_size: # only generate seed dataset
                break
            if len(seed_dataset) >= target_size and len(error_dataset) >= target_size: # generating both seed and error seed dataset
                break

            for problem_type in problem_types:
                if problem_type == 'code_f': # skip code_f dataset, we will generate it later
                    continue
                if problem_type == 'code_e' and len(error_dataset) >= target_size:
                    continue
                if problem_type != 'code_e' and len(seed_dataset) >= target_size:
                    continue
                seed_dataloader = self._create_train_code_gen_dataloader(
                    problem_type=problem_type,
                    data_len=self.config.data.train_batch_size,
                    dataset_key='error_seed' if problem_type == 'code_e' else 'seed',
                    seeding=True,
                )
                for batch_dict in seed_dataloader:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': True,
                        'validate': False,
                    }

                    # pad to be divisible by dp_size
                    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                    output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                    pad_size *= self.config.actor_rollout_ref.rollout.n

                    # unpad
                    output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

                    # If we're doing multiple samples, repeat the original batch
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    batch = batch.union(output_gen_batch)

                    # Store generated outputs
                    output_ids = batch.batch['responses']
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    local_entries = []
                    local_error_entries = []
                    for output_text in output_texts:
                        success, result = parse_code_input_output(
                            output_text,
                            parse_output=False,
                            remove_after_return=self.config.azr.reward.generation_reward_config.remove_after_return,
                            remove_comments=self.config.azr.reward.generation_reward_config.remove_comments,
                            remove_print=self.config.azr.reward.generation_reward_config.remove_print,
                            reject_multiple_functions=self.config.azr.reward.generation_reward_config.reject_multiple_functions,
                            f_replace_location=self.config.azr.reward.generation_reward_config.f_replace_location,
                            reject_test_input_in_code=self.config.azr.reward.generation_reward_config.reject_test_input_in_code,
                            code_location=self.config.azr.reward.generation_reward_config.code_location,
                        )
                        if success:
                            code_validity, output = self._executor.check_all(
                                code=result['code'],
                                inputs=result['input'],
                                banned_keywords=self.config.azr.data_selection_strategy.banned_words,
                                check_determinism=True,
                                imports=list(set(result['imports'])),
                                check_error=problem_type == 'code_e',
                                banned_keywords_for_errors_and_exceptions=self.config.azr.data_selection_strategy.banned_keywords_for_errors_and_exceptions,
                            )
                            if code_validity:
                                if problem_type == 'code_e':
                                    local_error_entries.append(
                                        {
                                            'snippet': result['code'],
                                            'input': result['input'],
                                            'output': output,
                                            'imports': result['imports'],
                                            'original_snippet': result['code'],
                                            'composite_functions': []
                                        }
                                    )
                                else:
                                    local_entries.append(
                                        {
                                            'snippet': result['code'],
                                            'input': result['input'],
                                            'output': output,
                                            'imports': result['imports'],
                                            'original_snippet': result['code'],
                                            'composite_functions': []
                                        }
                                    )

                    if self.config.azr.data_selection_strategy.get('generate_seed_dataset_only', False):
                        with open(self.config.azr.data_selection_strategy.output_seed_path.replace('.jsonl', f'_temp.jsonl'), 'a') as f:
                            for entry in local_entries:
                                f.write(json.dumps(entry) + '\n')

                    break # only use the first batch, to continuously generate more diverse data

                # Atomically update shared dataset
                if problem_type != 'code_e':
                    # Process locally first
                    processed_entries = process_elements(local_entries)
                    # Then send to ray
                    added_count = ray.get(
                        self.dataset_manager.update_seed.remote(processed_entries)
                    )
                    # Get updated dataset
                    seed_dataset = ray.get(self.dataset_manager.get_dataset.remote('seed'))
                    PrettyPrinter.status(
                        "WORKER", 
                        f"Added {added_count} new entries (Total: {len(seed_dataset)})", 
                        "info"
                    )
                    PrettyPrinter.progress_bar(
                        current=len(seed_dataset),
                        total=target_size,
                        label="Dataset Growth"
                    )
                    # Early exit if we've reached target size
                    if len(seed_dataset) >= target_size:
                        break

                elif problem_type == 'code_e':
                    # Process locally first
                    processed_entries = process_elements(local_error_entries)
                    # Then send to ray
                    error_added_count = ray.get(
                        self.dataset_manager.update_error_seed.remote(processed_entries)
                    )
                    error_dataset = ray.get(self.dataset_manager.get_dataset.remote('error_seed'))
                    PrettyPrinter.status(
                        "WORKER", 
                        f"Added {error_added_count} new entries (Total: {len(error_dataset)})", 
                        "info"
                    )
                    PrettyPrinter.progress_bar(
                        current=len(error_dataset),
                        total=target_size,
                        label="Error Dataset Growth"
                    )
                    if len(error_dataset) >= target_size:
                        break

        # now get the code_f dataset
        if 'code_f' in problem_types:
            code_f_dataset = []
            all_snippets = ray.get(self.dataset_manager.get_snippets.remote())
            while len(code_f_dataset) < target_size:
                # randomly sample a snippet from all_snippets
                code_f_dataset = ray.get(self.dataset_manager.get_dataset.remote('problem'))
                code_f_seed_dataloader = self._create_train_code_gen_dataloader(
                    data_len=len(all_snippets),
                    problem_type='code_f',
                    seeding=True
                )
                epoch_entries = []
                for batch in code_f_seed_dataloader:
                    batch: DataProto = DataProto.from_single_dict(batch)
                    gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'recompute_log_prob': False,
                        'do_sample': True,
                        'validate': False,
                    }

                    # pad to be divisible by dp_size
                    gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
                    output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                    pad_size *= self.config.actor_rollout_ref.rollout.n

                    # unpad
                    output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

                    # If we're doing multiple samples, repeat the original batch
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    batch = batch.union(output_gen_batch)

                    # Store generated outputs
                    output_ids = batch.batch['responses']
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    for idx, output_text in enumerate(output_texts):
                        success, result = parse_inputs_message(output_text, num_inputs=self.config.azr.data_selection_strategy.num_inputs)
                        if success:
                            outputs = []
                            for ipt in result['inputs']:
                                code_validity, output = self._executor.check_all(
                                    code=batch.non_tensor_batch['extra_info'][idx]['chosen_references'][0]['snippet'],
                                    inputs=ipt,
                                    banned_keywords=[],
                                    check_determinism=True,
                                    imports=batch.non_tensor_batch['extra_info'][idx]['imports'],
                                    check_error=False,
                                    banned_keywords_for_errors_and_exceptions=[],
                                )
                                outputs.append(output)
                                if not code_validity:
                                    break
                            if code_validity:
                                epoch_entries.append(
                                    {
                                        'snippet': batch.non_tensor_batch['extra_info'][idx]['chosen_references'][0]['snippet'],
                                        'inputs': result['inputs'],
                                        'outputs': outputs,
                                        'message': result['message'],
                                        'imports': batch.non_tensor_batch['extra_info'][idx]['imports'].tolist(),
                                    }
                                )

                # Then send to ray
                processed_entries = process_elements(epoch_entries)
                added_count = ray.get(self.dataset_manager.add_problem_batch.remote(processed_entries, self.global_steps))
                # Get updated dataset
                code_f_dataset = ray.get(self.dataset_manager.get_dataset.remote('problem'))
                PrettyPrinter.status(
                    "WORKER", 
                    f"Added {added_count} new entries (Total: {len(code_f_dataset)})", 
                    "info"
                )
                PrettyPrinter.progress_bar(
                    current=len(code_f_dataset),
                    total=target_size,
                    label="Code F Dataset Growth"
                )
                if self.config.azr.data_selection_strategy.get('generate_seed_dataset_only', False):
                    with open(self.config.azr.data_selection_strategy.output_code_f_seed_path.replace('.jsonl', f'_temp.jsonl'), 'a') as f:
                        for entry in code_f_dataset:
                            f.write(json.dumps(entry) + '\n')
                # Early exit if we've reached target size
                if len(code_f_dataset) >= target_size:
                    break

        # truncate the dataset to the target size
        ray.get(self.dataset_manager.truncate_datasets.remote(target_size, 'seed'))
        ray.get(self.dataset_manager.truncate_datasets.remote(target_size, 'error_seed'))
        ray.get(self.dataset_manager.truncate_datasets.remote(target_size, 'problem'))

        # Sample type statistics after seed initialization
        if self.global_steps == 0:  # Only log this on first initialization
            type_stats = ray.get(self.dataset_manager.get_all_type_statistics.remote())
            PrettyPrinter.section_header("Initial Type Statistics")
            for category, type_data in type_stats.items():
                category_display = {
                    'input_types': 'Input Types',
                    'output_types': 'Output Types',
                    'error_types': 'Error Types'
                }.get(category, category)

                if type_data:  # Only show if we have data
                    PrettyPrinter.status(category_display.upper(), f"Total types: {len(type_data)}", "info")
                    for type_name, stats in sorted(type_data.items(), 
                                                key=lambda x: x[1]['total_unique'], 
                                                reverse=True)[:5]:  # Show top 5 by unique count
                        PrettyPrinter.status(
                            f"  {type_name}", 
                            f"Unique: {stats['total_unique']}, Total: {stats['total_count']}", 
                            "info"
                        )
                        if stats['examples']:
                            example = stats['examples'][0]
                            if len(example) > 100:
                                example = example[:97] + "..."
                            PrettyPrinter.status("    Example", example, "info")

        # Final dataset from coordinator
        seed_dataset = ray.get(self.dataset_manager.get_dataset.remote('seed'))
        error_dataset = ray.get(self.dataset_manager.get_dataset.remote('error_seed'))
        code_f_dataset = ray.get(self.dataset_manager.get_dataset.remote('problem'))

        # Modify dataset saving condition
        if ('code_i' in problem_types or 'code_o' in problem_types) and self.config.azr.output_seed_path is not None:
            PrettyPrinter.status("DATASET", "Writing seed dataset to JSONL file...", "info")
            output_path = Path(self.config.azr.output_seed_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                for item in seed_dataset:
                    f.write(json.dumps(item) + '\n')
            PrettyPrinter.status("DATASET", f"Saved {len(seed_dataset)} entries to {str(output_path)}", "success")

        if 'code_e' in problem_types and self.config.azr.output_error_seed_path is not None:
            PrettyPrinter.status("DATASET", "Writing error seed dataset to JSONL file...", "info")
            output_path = Path(self.config.azr.output_error_seed_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for item in error_dataset:
                    f.write(json.dumps(item) + '\n')
            PrettyPrinter.status("DATASET", f"Saved {len(error_dataset)} entries to {str(output_path)}", "success")

        if 'code_f' in problem_types and self.config.azr.output_code_f_seed_path is not None:
            PrettyPrinter.status("DATASET", "Writing code f seed dataset to JSONL file...", "info")
            output_path = Path(self.config.azr.output_code_f_seed_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for item in code_f_dataset:
                    try:
                        f.write(json.dumps(item) + '\n')
                    except:
                        print(item)
                        raise Exception("Failed to save code f dataset")
            PrettyPrinter.status("DATASET", f"Saved {len(code_f_dataset)} entries to {str(output_path)}", "success")

        # Show a few sample entries
        if 'code_i' in problem_types or 'code_o' in problem_types:
            # Print detailed dataset summary
            PrettyPrinter.section_header("Seed Dataset Summary")
            PrettyPrinter.table(
                ["Key", "Value"],
                [
                    ["Total Samples", len(seed_dataset)],
                    ["Target Size", target_size],
                    ["Storage Path", self.config.azr.output_seed_path],
                    ["Sample Types", len(set(item['snippet'] for item in seed_dataset))],
                    ["Average Snippet Length", sum(len(item['snippet']) for item in seed_dataset) // len(seed_dataset) if seed_dataset else 0]
                ],
                title="Dataset Statistics"
            )

            PrettyPrinter.section_header("Sample Entries")
            # sample 3 entries
            for i, item in enumerate(random.sample(seed_dataset, self.config.azr.random_print_max_programs)):  
                PrettyPrinter.code_block(item['snippet'], "python")
                PrettyPrinter.status("INPUT", item['input'], "info")
                PrettyPrinter.status("OUTPUT", item['output'], "info")
                if i < 2:  # Don't print separator after last item
                    print("\n" + "-" * 80 + "\n")

        if 'code_e' in problem_types:
            PrettyPrinter.section_header("Error Dataset Summary")
            PrettyPrinter.table(
                ["Key", "Value"],
                [
                    ["Total Samples", len(error_dataset)],
                    ["Target Size", target_size],
                    ["Storage Path", self.config.azr.output_error_seed_path],
                    ["Sample Types", len(set(item['snippet'] for item in error_dataset))],
                    ["Average Snippet Length", sum(len(item['snippet']) for item in error_dataset) // len(error_dataset) if error_dataset else 0]
                ],
                title="Error Dataset Statistics"
            )
            PrettyPrinter.section_header("Error Sample Entries")
            # sample 3 entries
            for i, item in enumerate(random.sample(error_dataset, self.config.azr.random_print_max_programs)):  
                PrettyPrinter.code_block(item['snippet'], "python")
                PrettyPrinter.status("INPUT", item['input'], "info")
                PrettyPrinter.status("OUTPUT", item['output'], "info")
                if i < 2:  # Don't print separator after last item
                    print("\n" + "-" * 80 + "\n")
        
        if 'code_f' in problem_types:
            PrettyPrinter.section_header("Code F Dataset Summary")
            PrettyPrinter.table(
                ["Key", "Value"],
                [
                    ["Total Samples", len(code_f_dataset)],
                    ["Target Size", target_size],
                    ["Storage Path", self.config.azr.output_code_f_seed_path],
                    ["Sample Types", len(set(item['snippet'] for item in code_f_dataset))],
                    ["Average Snippet Length", sum(len(item['snippet']) for item in code_f_dataset) // len(code_f_dataset) if code_f_dataset else 0]
                ],
                title="Code F Dataset Statistics"
            )

            PrettyPrinter.section_header("Code F Sample Entries")
            # sample 3 entries
            for i, item in enumerate(random.sample(code_f_dataset, self.config.azr.random_print_max_programs)):  
                PrettyPrinter.code_block(item['snippet'], "python")
                PrettyPrinter.status("INPUTS", item['inputs'], "info")
                PrettyPrinter.status("OUTPUTS", item['outputs'], "info")
                PrettyPrinter.status("MESSAGE", item['message'], "info")
                if i < 2:  # Don't print separator after last item
                    print("\n" + "-" * 80 + "\n")

        return seed_dataset, error_dataset, code_f_dataset

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

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

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # base model chat template
        if self.config.actor_rollout_ref.model.pretrained_tokenizer:
            self.tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True) and self.global_steps == 0:
            PrettyPrinter.section_header(f"Starting Initial Validation")
            val_metrics = self._validate()
            PrettyPrinter.table(
                ["Metric", "Value"],
                [[k, v] for k, v in val_metrics.items()],
                title="Initial Validation Metrics"
            )
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        if self.loaded_datasets:
            PrettyPrinter.section_header(f"Resuming training from checkpoint")
            # print the lengths of the datasets
            for dataset_name in ['input', 'output', 'error', 'input_steps', 'output_steps', 'error_steps', 'input_steps_counter', 'output_steps_counter', 'error_steps_counter']:
                PrettyPrinter.status("DATA", f"Length of {dataset_name}: {ray.get(self.dataset_manager.get_dataset_size.remote(dataset_name))}", "info")
        else:
            PrettyPrinter.section_header(f"Creating initial seed datasets")
            # create init dataset
            need_seed_dataset = any(problem_type != 'code_e' for problem_type in self.config.azr.problem_types) or 'code_f' in self.config.azr.problem_types
            need_error_dataset = 'code_e' in self.config.azr.problem_types
            need_code_f_dataset = 'code_f' in self.config.azr.problem_types

            # Initialize with defaults
            seed_dataset = []
            error_dataset = []
            code_f_dataset = []

            # Load or generate seed dataset if needed
            if need_seed_dataset:
                if self.config.azr.seed_dataset is not None:
                    PrettyPrinter.status("DATA", "Loading seed dataset from file...", "info")
                    with open(self.config.azr.seed_dataset, 'r') as file:
                        seed_dataset = [json.loads(line) for line in file]
                    seed_dataset = seed_dataset[:self.config.azr.data_selection_strategy.data_len * 
                                                self.config.azr.data_selection_strategy.seed_batch_factor]
                    PrettyPrinter.status("DATA", f"Loaded {len(seed_dataset)} seed entries", "success")
                    if 'code_f' in self.config.azr.problem_types: # we need seed to generate code_f
                        ray.get(self.dataset_manager.update_seed.remote(seed_dataset))
                else:
                    PrettyPrinter.status("DATA", "Seed dataset not provided, will generate", "info")

            # Load or prepare to generate error dataset if needed
            if need_error_dataset:
                if self.config.azr.error_seed_dataset is not None:
                    PrettyPrinter.status("DATA", "Loading error seed dataset from file...", "info")
                    with open(self.config.azr.error_seed_dataset, 'r') as file:
                        error_dataset = [json.loads(line) for line in file]
                    error_dataset = error_dataset[:self.config.azr.data_selection_strategy.data_len * 
                                                self.config.azr.data_selection_strategy.seed_batch_factor]
                    PrettyPrinter.status("DATA", f"Loaded {len(error_dataset)} error entries", "success")
                else:
                    PrettyPrinter.status("DATA", "Error seed dataset not provided, will generate", "info")

            if need_code_f_dataset:
                if self.config.azr.code_f_seed_dataset is not None:
                    PrettyPrinter.status("DATA", "Loading code f seed dataset from file...", "info")
                    with open(self.config.azr.code_f_seed_dataset, 'r') as file:
                        code_f_dataset = [json.loads(line) for line in file]
                    code_f_dataset = code_f_dataset[:self.config.azr.data_selection_strategy.data_len * 
                                                self.config.azr.data_selection_strategy.seed_batch_factor]
                    PrettyPrinter.status("DATA", f"Loaded {len(code_f_dataset)} code f entries", "success")

            # Generate missing datasets if needed
            need_to_generate_seed = need_seed_dataset and len(seed_dataset) == 0
            need_to_generate_error = need_error_dataset and len(error_dataset) == 0
            need_to_generate_code_f = need_code_f_dataset and len(code_f_dataset) == 0

            if need_to_generate_seed or need_to_generate_error or need_to_generate_code_f:
                sample_problem_types = []
                for problem_type in self.config.azr.problem_types:
                    if problem_type == 'code_e' and need_to_generate_error:
                        sample_problem_types.append(problem_type)
                    elif problem_type != 'code_e' and need_to_generate_seed:
                        sample_problem_types.append(problem_type)
                    elif problem_type == 'code_f' and need_to_generate_code_f:
                        sample_problem_types.append(problem_type)
                PrettyPrinter.status("DATA", f"Generating missing datasets for {', '.join(sample_problem_types)}...", "info")
                generated_seed, generated_error, generated_code_f = self._init_seed_dataset(problem_types=sample_problem_types)

                if need_to_generate_seed:
                    seed_dataset = generated_seed
                    PrettyPrinter.status("DATA", f"Generated {len(seed_dataset)} seed entries", "success")

                if need_to_generate_error:
                    error_dataset = generated_error
                    PrettyPrinter.status("DATA", f"Generated {len(error_dataset)} error entries", "success")
                
                if need_to_generate_code_f:
                    code_f_dataset = generated_code_f
                    PrettyPrinter.status("DATA", f"Generated {len(code_f_dataset)} code f entries", "success")

                if self.config.azr.get('generate_seed_dataset_only', False):
                    return

            # Now initialize datasets in dataset manager
            if need_seed_dataset:
                assert len(seed_dataset) >= self.config.azr.data_selection_strategy.data_len

                if 'code_i' in self.config.azr.problem_types:
                    # Process locally first
                    processed_seed_dataset = process_elements(seed_dataset)
                    # Initialize input dataset with seed data
                    ray.get(self.dataset_manager.add_input_batch.remote(processed_seed_dataset, self.global_steps))
                    PrettyPrinter.status(
                        "DATA", 
                        f"Input dataset initialized with {len(seed_dataset)} entries", 
                        "success"
                    )

                if 'code_o' in self.config.azr.problem_types:
                    processed_seed_dataset = process_elements(seed_dataset)
                    ray.get(self.dataset_manager.add_output_batch.remote(processed_seed_dataset, self.global_steps))
                    PrettyPrinter.status(
                        "DATA", 
                        f"Output dataset initialized with {len(seed_dataset)} entries", 
                        "success"
                    )

            if need_error_dataset:
                assert len(error_dataset) >= self.config.azr.data_selection_strategy.data_len
                processed_error_dataset = process_elements(error_dataset)
                ray.get(self.dataset_manager.add_error_batch.remote(processed_error_dataset, self.global_steps))
                PrettyPrinter.status(
                    "DATA", 
                    f"Error dataset initialized with {len(error_dataset)} entries", 
                    "success"
                )

            if need_code_f_dataset:
                assert len(code_f_dataset) >= self.config.azr.data_selection_strategy.data_len
                processed_code_f_dataset = process_elements(code_f_dataset)
                ray.get(self.dataset_manager.add_problem_batch.remote(processed_code_f_dataset, self.global_steps))
                PrettyPrinter.status(
                    "DATA", 
                    f"Code f dataset initialized with {len(code_f_dataset)} entries", 
                    "success"
                )

        # we start from step 1
        self.global_steps += 1
        if self.config.azr.pretrain_pred_steps > 0 and self.global_steps <= self.config.azr.pretrain_pred_steps:
            self.pretrain_pred = True
        else:
            self.pretrain_pred = False

        while self.global_steps < self.total_training_steps:
            PrettyPrinter.section_header(f"Training Step {self.global_steps}")
            if self.config.azr.data_selection_strategy.composite_scheduler.enabled:
                self.scheduler_step()

            PrettyPrinter.progress_bar(
                current=self.global_steps,
                total=self.total_training_steps,
                label="Training Progress"
            )

            data_len = self.config.data.train_batch_size * self.config.azr.data_selection_strategy.update_iteration
            if 'code_i' in self.config.azr.problem_types:
                gen_code_i_dataloader = self._create_train_code_gen_dataloader(
                    problem_type='code_i',
                    data_len=data_len,
                )
                pred_code_i_dataloader = self._create_train_code_pred_dataloader(
                    problem_type='code_i',
                    data_len=data_len,
                )
            if 'code_o' in self.config.azr.problem_types:
                gen_code_o_dataloader = self._create_train_code_gen_dataloader(
                    problem_type='code_o',
                    data_len=data_len,
                )
                pred_code_o_dataloader = self._create_train_code_pred_dataloader(
                    problem_type='code_o',
                    data_len=data_len,
                )

            if 'code_e' in self.config.azr.problem_types:
                gen_code_e_dataloader = self._create_train_code_gen_dataloader(
                    problem_type='code_e',
                    data_len=data_len,
                )
                pred_code_e_dataloader = self._create_train_code_pred_dataloader(
                    problem_type='code_e',
                    data_len=data_len,
                )

            if 'code_f' in self.config.azr.problem_types:
                gen_code_f_dataloader = self._create_train_code_gen_dataloader(
                    data_len=data_len,
                    problem_type='code_f',
                    seeding=False,
                )
                pred_code_f_dataloader = self._create_train_code_pred_dataloader(
                    problem_type='code_f',
                    data_len=data_len,
                )
            for _ in range(self.config.azr.data_selection_strategy.update_iteration):
                metrics = {}
                timing_raw = {}
                batches = {}
                with marked_timer('step', timing_raw):
                    # Clean up executor periodically
                    if self.global_steps - self._last_cleanup_step >= self._cleanup_frequency:
                        PrettyPrinter.section_header("Periodic Cleanup")
                        with marked_timer('cleanup', timing_raw):
                            self.cleanup()
                        self._last_cleanup_step = self.global_steps

                    if 'code_i' in self.config.azr.problem_types:
                        if not self.pretrain_pred:
                            batch_dict = next(gen_code_i_dataloader)
                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                            batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='gen_code_i', executor=self._executor)
                            if self.config.azr.train_propose:
                                batches[f'gen_code_i'] = batch
                        batch_dict = next(pred_code_i_dataloader)
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='pred_code_i', executor=self._executor)
                        batches[f'pred_code_i'] = batch

                    if 'code_o' in self.config.azr.problem_types:
                        if not self.pretrain_pred:
                            batch_dict = next(gen_code_o_dataloader)
                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                            batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='gen_code_o', executor=self._executor)
                            if self.config.azr.train_propose:
                                batches[f'gen_code_o'] = batch
                        batch_dict = next(pred_code_o_dataloader)
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='pred_code_o', executor=self._executor)
                        batches[f'pred_code_o'] = batch

                    if 'code_e' in self.config.azr.problem_types:
                        if not self.pretrain_pred:
                            batch_dict = next(gen_code_e_dataloader)
                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                            batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='gen_code_e', executor=self._executor)
                            if self.config.azr.train_propose:
                                batches[f'gen_code_e'] = batch
                        batch_dict = next(pred_code_e_dataloader)
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='pred_code_e', executor=self._executor)
                        batches[f'pred_code_e'] = batch

                    if 'code_f' in self.config.azr.problem_types:
                        if not self.pretrain_pred:
                            batch_dict = next(gen_code_f_dataloader)
                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                            batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='gen_code_f', executor=self._executor)
                            if self.config.azr.train_propose:
                                batches[f'gen_code_f'] = batch
                        batch_dict = next(pred_code_f_dataloader)
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        batch, metrics = self._compute_batch(batch, metrics, timing_raw, problem_type='pred_code_f', executor=self._executor)
                        batches[f'pred_code_f'] = batch

                    # concatenate batches
                    batch = DataProto.concat(list(batches.values()))

                    PrettyPrinter.section_header(f"Starting Parameter Updates")
                    # update critic
                    if self.use_critic:
                        with marked_timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    PrettyPrinter.section_header(f"Starting Validation")
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            PrettyPrinter.table(
                                ["Data Source", "Average Score"],
                                [[k, v] for k, v in val_metrics.items()],
                                title="Validation Results"
                            )
                        metrics.update(val_metrics)

                    # print the statistics of the number of programs in the dataset
                    if 'code_i' in self.config.azr.problem_types:
                        PrettyPrinter.status(
                            "DATA", 
                            f"Number of programs in the input dataset: {ray.get(self.dataset_manager.get_dataset_size.remote('input'))}", 
                            "info"
                        )
                    if 'code_o' in self.config.azr.problem_types:
                        PrettyPrinter.status(
                            "DATA", 
                            f"Number of programs in the output dataset: {ray.get(self.dataset_manager.get_dataset_size.remote('output'))}", 
                            "info"
                        )
                    if 'code_e' in self.config.azr.problem_types:
                        PrettyPrinter.status(
                            "DATA", 
                            f"Number of programs in the error dataset: {ray.get(self.dataset_manager.get_dataset_size.remote('error'))}", 
                            "info"
                        )
                    if 'code_f' in self.config.azr.problem_types:
                        PrettyPrinter.status(
                            "DATA", 
                            f"Number of programs in the code_f dataset: {ray.get(self.dataset_manager.get_dataset_size.remote('problem'))}", 
                            "info"
                        )
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics, separate problem types
                all_types = []
                if 'code_i' in self.config.azr.problem_types:
                    if not self.pretrain_pred:
                        all_types.append('gen_code_i')
                    all_types.append('pred_code_i')
                if 'code_o' in self.config.azr.problem_types:
                    if not self.pretrain_pred:
                        all_types.append('gen_code_o')
                    all_types.append('pred_code_o')
                if 'code_e' in self.config.azr.problem_types:
                    if not self.pretrain_pred:
                        all_types.append('gen_code_e')
                    all_types.append('pred_code_e')
                if 'code_f' in self.config.azr.problem_types:
                    if not self.pretrain_pred:
                        all_types.append('gen_code_f')
                    all_types.append('pred_code_f')
                sep_batches = batch.chunk(len(all_types))
                for sep_batch, problem_type in zip(sep_batches, all_types):
                    sep_metrics = compute_data_metrics(batch=sep_batch, use_critic=self.use_critic, tokenizer=self.tokenizer)
                    sep_metrics = {f'{problem_type}/{k}': v for k, v in sep_metrics.items()}
                    metrics.update(sep_metrics)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # Get and log type statistics periodically
                type_stats = ray.get(self.dataset_manager.get_all_type_statistics.remote())

                # Log summary metrics about types
                for category, type_data in type_stats.items():
                    # Calculate diversity metrics
                    total_types = len(type_data)
                    total_unique_values = sum(stats["total_unique"] for stats in type_data.values())
                    total_instances = sum(stats["total_count"] for stats in type_data.values())

                    # Add to metrics
                    metrics[f"types/{category}/distinct_types"] = total_types
                    metrics[f"types/{category}/total_unique_values"] = total_unique_values
                    metrics[f"types/{category}/total_instances"] = total_instances

                    # Per-type metrics
                    for type_name, stats in type_data.items():
                        metrics[f"types/{category}/{type_name}/unique_count"] = stats["total_unique"]
                        metrics[f"types/{category}/{type_name}/total_count"] = stats["total_count"]

                # Print type statistics summary
                PrettyPrinter.section_header("Type Statistics Summary")
                for category, type_data in type_stats.items():
                    category_display = {
                        'input_types': 'Input Types',
                        'output_types': 'Output Types',
                        'error_types': 'Error Types',
                    }.get(category, category)

                    PrettyPrinter.status(category_display.upper(), f"Total types: {len(type_data)}", "info")
                    for type_name, stats in sorted(type_data.items(), 
                                                    key=lambda x: x[1]['total_unique'], 
                                                    reverse=True)[:5]:  # Show top 5 by unique count
                        PrettyPrinter.status(
                            f"  {type_name}", 
                            f"Unique: {stats['total_unique']}, Total: {stats['total_count']}", 
                            "info"
                        )
                        if stats['examples']:
                            example = stats['examples'][0]
                            if len(example) > 100:
                                example = example[:97] + "..."
                            PrettyPrinter.status("    Example", example, "info")

                PrettyPrinter.table(
                    ["Category", "Value"],
                    [[k, v] for k, v in metrics.items()],
                    title="Step Metrics"
                )

                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps >= self.config.azr.pretrain_pred_steps:
                    self.pretrain_pred = False

                self.global_steps += 1

                gc.collect()

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        PrettyPrinter.section_header(f"Starting Final Validation")
                        val_metrics = self._validate()
                        PrettyPrinter.table(
                            ["Data Source", "Average Score"],
                            [[k, v] for k, v in val_metrics.items()],
                            title="Final Validation Results"
                        )
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with marked_timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return

    def _validate(self):
        """
        The validation loop of PPO.
        The only difference is logging more metrics.
        """
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

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            PrettyPrinter.status("VALID", "Generation completed", "success")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, eval_metrics, _, _ = self.val_reward_fn(
                test_batch,
                problem_type=None,
                executor=self._executor,
            )
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

        return metric_dict

    def _save_datasets(self, save_dir: Path):
        """Save input/output datasets as JSONL files"""
        save_dir.mkdir(parents=True, exist_ok=True)

        # get all datasets and type counters
        datasets_with_types = ray.get(self.dataset_manager.get_all_data_with_type_counters.remote())

        # save datasets
        pickle.dump(datasets_with_types, open(save_dir / 'datasets.pkl', 'wb'))
        PrettyPrinter.status("SAVE", f"Saved datasets and type counters to {save_dir}", "success")

    def _load_datasets(self, save_dir: Path):
        """Load input/output datasets from JSONL files"""
        datasets_with_types = pickle.load(open(Path(save_dir) / 'datasets' / 'datasets.pkl', 'rb'))

        # Filter datasets based on global step
        if self.global_steps > 0:
            # Filter datasets that have step info
            for dataset_key in ['input', 'output', 'error', 'problem']:
                steps_key = f"{dataset_key}_steps"
                if steps_key in datasets_with_types and dataset_key in datasets_with_types:
                    # Create lists of entries to keep
                    filtered_data = []
                    filtered_steps = []

                    # Only keep entries with steps less than current global_steps
                    for entry, step in zip(datasets_with_types[dataset_key], datasets_with_types[steps_key]):
                        if step <= self.global_steps:
                            filtered_data.append(entry)
                            filtered_steps.append(step)

                    # Update the datasets
                    datasets_with_types[dataset_key] = filtered_data
                    datasets_with_types[steps_key] = filtered_steps

                    # Also filter the step counter dictionaries
                    counter_key = f"{dataset_key}_steps_counter"
                    if counter_key in datasets_with_types:
                        filtered_counter = defaultdict(int)
                        for step, count in datasets_with_types[counter_key].items():
                            if step <= self.global_steps:
                                filtered_counter[step] = count
                        datasets_with_types[counter_key] = filtered_counter

            PrettyPrinter.status("FILTER", f"Filtered datasets to only include entries with steps <= {self.global_steps}", "info")

        ray.get(self.dataset_manager.full_load_data_with_type_counters.remote(datasets_with_types))
        PrettyPrinter.status("LOAD", f"Loaded datasets and type counters from {self.config.trainer.default_local_dir}", "success")
        self.loaded_datasets = True

    def _save_checkpoint(self):
        super()._save_checkpoint()
        # save datasets
        self._save_datasets(Path(self.config.trainer.default_local_dir) / 'datasets')
        PrettyPrinter.status("SAVE", f"Saved checkpoint to {self.config.trainer.default_local_dir}", "success")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps == 0:
            PrettyPrinter.section_header(f"Training from scratch")
        else:
            PrettyPrinter.section_header(f"Resuming training from checkpoint, step {self.global_steps}")

        # load datasets
        # first check if all the datasets exist
        code_dir = Path(self.config.trainer.default_local_dir) / 'code'
        self._code_dir = code_dir
        self.loaded_datasets = False
        if self.config.trainer.resume_mode == 'auto' and os.path.exists(os.path.join(self.config.trainer.default_local_dir, 'datasets', 'datasets.pkl')):
            self._load_datasets(self.config.trainer.default_local_dir)
        elif self.config.trainer.resume_mode == 'disable':
            if code_dir.exists():
                # delete all files and subdirectories in the code_dir
                for file in code_dir.glob('**/*'):
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        file.rmdir()
            PrettyPrinter.status("Directory", f"Cleaned existing code directory at {code_dir}", "info")
        elif not code_dir.exists():
            code_dir.mkdir(parents=True, exist_ok=True)
            PrettyPrinter.status("Directory", f"Created new code directory at {code_dir}", "info")

    def scheduler_step(self):
        if self.config.azr.data_selection_strategy.composite_scheduler.enabled:
            # Update number of programs - calculate directly based on global steps
            if self.global_steps >= self.config.azr.data_selection_strategy.composite_scheduler.update_num_programs_start:
                steps_since_start = self.global_steps - self.config.azr.data_selection_strategy.composite_scheduler.update_num_programs_start
                num_updates = steps_since_start // self.config.azr.data_selection_strategy.composite_scheduler.update_num_programs_interval

                # Calculate new value directly from initial value + increments
                initial_max = self.config.azr.data_selection_strategy.max_programs_initial
                new_max = min(initial_max + num_updates, self.config.azr.data_selection_strategy.composite_scheduler.num_programs_max)

                # Only log if value changed
                if new_max != self.config.azr.data_selection_strategy.composite_function_n_max:
                    current_max = self.config.azr.data_selection_strategy.composite_function_n_max
                    self.config.azr.data_selection_strategy.composite_function_n_max = new_max
                    PrettyPrinter.status("Scheduler", f"Updated max programs from {current_max} to {new_max}", "info")

            # Update composite probability - calculate directly based on global steps
            if self.global_steps >= self.config.azr.data_selection_strategy.composite_scheduler.update_probability_start:
                steps_since_start = self.global_steps - self.config.azr.data_selection_strategy.composite_scheduler.update_probability_start
                num_updates = steps_since_start // self.config.azr.data_selection_strategy.composite_scheduler.update_probability_interval

                # Calculate new value directly from initial value + increments
                initial_prob = self.config.azr.data_selection_strategy.composite_chance_initial
                new_prob = min(initial_prob + (num_updates * self.config.azr.data_selection_strategy.composite_scheduler.update_probability_increment),
                              self.config.azr.data_selection_strategy.composite_scheduler.update_probability_max)

                # Only log if value changed
                if new_prob != self.config.azr.data_selection_strategy.composite_chance:
                    current_prob = self.config.azr.data_selection_strategy.composite_chance
                    self.config.azr.data_selection_strategy.composite_chance = new_prob
                    PrettyPrinter.status("Scheduler", f"Updated composite probability from {current_prob:.2f} to {new_prob:.2f}", "info")
