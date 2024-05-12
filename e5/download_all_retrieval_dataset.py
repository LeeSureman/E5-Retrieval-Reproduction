import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB, AbsTaskRetrieval, DRESModel
from peft import PeftModel, PeftConfig

from utils import pool, logger, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, \
    create_batch_dict, my_create_batch_dict
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE


def main():


    task_names = [t.description["name"] for t in MTEB(task_types=['Retrieval'], task_langs=['en']).tasks]
    task_names = [t for t in task_names if t != 'MSMARCOv2']
    logger.info('Tasks: {}'.format(task_names))

    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=['en'])