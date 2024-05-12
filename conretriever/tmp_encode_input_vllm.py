import os.path

from transformers import AutoModel
import argparse
import jsonlines
from fastchat.model import get_conversation_template
import tqdm
# from utils import _setup_logger
# logger = _setup_logger()
from vllm import LLM, SamplingParams
import torch



if __name__ == '__main__':

    tmp = list(jsonlines.open('retrieval_data/multi_dataset_2024_2_13/nq.jsonl'))
    raw_data = []
    for tmp_js in tmp:
        pos = tmp_js['positive'][0]
        raw_data.append(pos)

    tensor_parallel_size = 1


    #VLLM related
    max_num_seqs = 256

    sampling_params = SamplingParams(1, temperature=0, top_p=0.95, max_tokens=1)
    llm = LLM(model='/remote-home/share/models/mistral_7b_instruct', tensor_parallel_size=tensor_parallel_size,
              max_model_len=1024,dtype='float16', gpu_memory_utilization=0.8 if tensor_parallel_size<=4 else 0.9,
              max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_seqs*128)

    outputs_vllm = llm.generate(raw_data, sampling_params)