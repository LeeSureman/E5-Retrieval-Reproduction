# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append('..')
sys.path.append('.')
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
# from transformers import Trainer
from dpo_trainer import DPO_Trainer, MyCallback
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from modeling_dpo import DPO_Dataset

from utils import _setup_logger

logger = _setup_logger()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class DPO_Arguments:
    reference_free: bool=field(default=False)
    dpo_beta: float=field(default=None)
    dpo_label_smoothing: float=field(default=0)
    name: str=field(default='dpo')
    sft_coef_when_dpo: float=field(default=None)
    sft_mode: bool=field(default=False)




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    dpo_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    eval_dpo_data_path: str=None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_prompt_length: int=None
    sequential_sampler: bool=True
    flash_attention: int = field(default=None)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


import jsonlines

def add_sft_template(user_input, sft_template_type):
    if sft_template_type == 'vicuna':
        conv_vicuna = get_conversation_template('vicuna')
        conv_vicuna.append_message(conv_vicuna.roles[0], user_input)
        result = conv_vicuna.get_prompt() + conv_vicuna.roles[1] + ':'
        return result
    else:
        raise NotImplementedError

from my_utils import get_hyde_prompt_template
hyde_prompt_dict = get_hyde_prompt_template()
import copy

def wrap_retrieval_query_into_full_prompt(query, task):
    hyde_prompt_template = hyde_prompt_dict[task]
    hyde_prompt = hyde_prompt_template.format(query)
    final_prompt = add_sft_template(hyde_prompt, 'vicuna')

    return final_prompt



def wrap_dpo_dict_data(prompt_to_pairs_dict):
    final_prompt_to_pairs_dict = copy.copy(prompt_to_pairs_dict)

    for prompt, tartget_info in prompt_to_pairs_dict.items():
        retrieval_query = prompt
        task = tartget_info['task_name']
        final_prompt = wrap_retrieval_query_into_full_prompt(retrieval_query, task)
        final_prompt_to_pairs_dict[final_prompt] = tartget_info
        del final_prompt_to_pairs_dict[retrieval_query]

    return final_prompt_to_pairs_dict


    pass



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DPO_Arguments)
    )
    model_args, data_args, training_args, dpo_args = parser.parse_args_into_dataclasses()
    if dpo_args.sft_mode:
        dpo_args.name = 'sft'
    local_rank = training_args.local_rank
    training_args.report_to = []

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    prompt_to_pairs_dict = json.load(open(data_args.dpo_data_path))
    prompt_to_pairs_dict = wrap_dpo_dict_data(prompt_to_pairs_dict)

    dpo_dataset = DPO_Dataset(prompt_to_pairs_dict, tokenizer, shuffle=True, max_length=training_args.model_max_length,
                              max_prompt_length=training_args.max_prompt_length, sft_mode=dpo_args.sft_mode, seed=training_args.seed)


    check_index=1
    if check_index>=0:
        tmp_example =dpo_dataset[check_index]
        rank0_print('input keys:{}'.format(tmp_example.keys()))
        # print(dpo_dataset[check_index])
        for k, v in tmp_example.items():
            rank0_print('{}:\n{}'.format(k, v))
            if 'attention_mask' in k:
                continue
            rank0_print('{}:'.format(k))
            rank0_print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(filter(lambda x:x>=0, v))))
            rank0_print('*'*30)

        # exit()

    if 'mistral' in model_args.model_name_or_path or training_args.flash_attention:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )

    # Load data
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    policy = model

    if dpo_args.name == 'dpo':
        rank0_print('building reference model')
        if 'mistral' in model_args.model_name_or_path or training_args.flash_attention:
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                trust_remote_code=model_args.trust_remote_code,
                # attn_implementation="flash_attention_2",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        else:
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=True
            )
        reference_model.eval()



    trainer = DPO_Trainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=dpo_dataset
    )
    my_callback = MyCallback(trainer)
    trainer.add_callback(my_callback)
    trainer.set_data_collator()
    if training_args.sequential_sampler:
        logger.info('set trainer.accelerator for sequential_sampler=True')
        trainer.accelerator.device_placement = True
        trainer.accelerator.dispatch_batches = True

    if dpo_args.name == 'dpo':
        rank0_print('move reference_model to cuda')
        reference_model = reference_model.cuda()
        trainer.reference_model = reference_model
    trainer.dpo_args = dpo_args
    rank0_print('dpo_args:\n{}'.format(dpo_args))
    rank0_print('model_args:\n{}'.format(model_args))

    # eval_dpo_data
    if data_args.eval_dpo_data_path != None:
        eval_dpo_data_dict = json.load(open(data_args.eval_dpo_data_path))
        rank0_print('eval data dict size: {}'.format(len(eval_dpo_data_dict)))
        eval_dpo_dataset = DPO_Dataset(eval_dpo_data_dict, tokenizer, shuffle=False,
                                  max_length=training_args.model_max_length,
                                  max_prompt_length=training_args.max_prompt_length, sft_mode=dpo_args.sft_mode,
                                  seed=training_args.seed)
        eval_dpo_dataset.update_data_for_run()
        # trainer.eval_dpo_data(eval_dpo_dataset)
    else:
        eval_dpo_dataset = None

    trainer.eval_dpo_dataset = eval_dpo_dataset
    # exit()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
