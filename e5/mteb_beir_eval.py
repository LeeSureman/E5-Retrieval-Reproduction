import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import jsonlines
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB, AbsTaskRetrieval, DRESModel
from peft import PeftModel, PeftConfig

from utils import pool, logger, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, \
    create_batch_dict, my_create_batch_dict
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE

import sys

sys.path.append('.')
sys.path.append('..')

from V6.replace_mistral_causal_lm_forward import replace_mistral_causal_lm_forward_func, replace_gemma_causal_lm_forward_func, \
replace_internlm2_causal_lm_forward_func

replace_mistral_causal_lm_forward_func()
replace_gemma_causal_lm_forward_func()

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model-name-or-path', default='intfloat/e5-small-v2',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--output-dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--doc-as-query', action='store_true',
                    help='use query prefix for passages, only used for Quora as it is a symmetric task')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prefix-type', default='query_or_passage', help='prefix type')
parser.add_argument('--dry-run', action='store_true', help='whether to run the script in dry run mode')
parser.add_argument('--representation_token_num', type=int, )
parser.add_argument('--model_dtype')
parser.add_argument('--task_names')
parser.add_argument('--query_max_length', required=True, type=int)
parser.add_argument('--passage_max_length', required=True, type=int)
parser.add_argument('--representation_id', type=int, required=True)
parser.add_argument('--wrap_q_p', choices=['instruction', 'query_or_passage'], required=True)
parser.add_argument('--force_retrieval_model', choices=['repllama', 'original', 'mine', 'default'])
parser.add_argument('--batch_per_gpu', type=int, default=24)
parser.add_argument('--pseudo_passage_fp',default='0')
parser.add_argument('--how_to_use_pseudo_passage',choices=['embedding_average','concat'])
parser.add_argument('--doc_embedding_cache_dir',default='None')
# parser.add_argument('--')

args = parser.parse_args()


if args.doc_embedding_cache_dir.lower() in ['none', '0']:
    args.doc_embedding_cache_dir = None
    logger.info('set args.doc_embedding_cache_dir to None')

base_name: str = args.model_name_or_path.split('/')[-1]
args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)




if 'model_repllama' in args.model_name_or_path:
    args.pool_type = 'last'
    args.prefix_type = 'query_or_passage'

print('pool_type:{}'.format(args.pool_type))
print('prefix_type:{}'.format(args.prefix_type))

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
assert args.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'
os.makedirs(args.output_dir, exist_ok=True)

with jsonlines.open('{}/my_logs.jsonl'.format(args.output_dir), 'w') as f_out:
    for k,v in args.__dict__.items():
        logger.info('{}: {}'.format(k,v))
        f_out.write({k:v})

class RepLLAMARetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.see_index = -1
        if args.model_dtype == 'fp16':
            model_dtype = torch.float16
        elif args.model_dtype == 'bf16':
            model_dtype = torch.bfloat16
        elif args.model_dtype == 'fp32':
            model_dtype = torch.float32
        else:
            raise NotImplementedError

        if os.path.exists('{}/adapter_config.json'.format(args.model_name_or_path)):

            peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
            if os.path.exists('/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/llama_2_7b_base'):
                base_model_path = '/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/llama_2_7b_base'
            elif os.path.exists('/remote-home/share/models/llama_v2_hf/7b'):
                base_model_path = '/remote-home/share/models/llama_v2_hf/7b'
            else:
                raise NotImplementedError

            logger.info('peft use base model at: {}'.format(base_model_path))

            base_model_config = AutoConfig.from_pretrained(base_model_path)
            base_model_config.output_hidden_states = False

            base_model = AutoModel.from_pretrained(base_model_path, torch_dtype=model_dtype,
                                                   config=base_model_config,
                                                   # attn_implementation="flash_attention_2",
                                                   )

            peft_model_id = args.model_name_or_path
            peft_config = PeftConfig.from_pretrained(peft_model_id)
            model = PeftModel.from_pretrained(base_model, peft_model_id)
            model = model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        else:
            base_model_config = AutoConfig.from_pretrained(args.model_name_or_path)
            base_model_config.output_hidden_states = False
            model = AutoModel.from_pretrained(args.model_name_or_path, config=base_model_config,
                                              torch_dtype=model_dtype)
            if 'mistral' in args.model_name_or_path:
                if os.path.exists('/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base'):
                    base_model_path = '/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base'
                elif os.path.exists('/remote-home/share/models/mistral_7b_base'):
                    base_model_path = '/remote-home/share/models/mistral_7b_base'
                else:
                    raise NotImplementedError
            else:
                if os.path.exists('/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/llama_2_7b_base'):
                    base_model_path = '/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/llama_2_7b_base'
                elif os.path.exists('/remote-home/share/models/llama_v2_hf/7b'):
                    base_model_path = '/remote-home/share/models/llama_v2_hf/7b'
                else:
                    raise NotImplementedError
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

            logger.info('llm_type: {}'.format(type(model)))
            logger.info('tokenizer: {}'.format(base_model_path))

        self.encoder = model

        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'right'

        # self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        if args.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
        else:
            input_texts = [self.prompt + q for q in queries]

        if self.see_index >= 0:
            print('original input query:\n{}'.format(input_texts[self.see_index]))

        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if args.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]
        if self.see_index >= 0:
            print('original input passage:\n{}'.format(input_texts[self.see_index]))

        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        if self.gpu_count == 8:
            batch_size = 20 * self.gpu_count
        else:
            batch_size = 24 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, always_add_eos=(args.pool_type == 'last'))

            # debug start

            if self.see_index >= 0:
                attn_mask = batch_dict['attention_mask'][self.see_index]
                is_not_pad = (attn_mask == 1)
                input_ids = batch_dict['input_ids'][self.see_index][is_not_pad]
                print('input_ids:\n{}'.format(input_ids))
                print()
                print(
                    'input_tokens:\n{}'.format(self.tokenizer.convert_ids_to_tokens(input_ids)))
                print()
                print('input_string:\n{}'.format(self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids))))
                print()

                exit()
            # debug finish

            # print('input_ids:{}'.format(batch_dict['input_ids'].size()))
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict, output_hidden_states=False)
                # print('outputs:{}'.format(outputs.keys()))
                # for k,v in outputs.items():
                #     print('{}:\n{}'.format(k,v))
                # exit()
                # print('outputs:{}')
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.args = args
        if args.pseudo_passage_fp != '0':
            print('use pseudo passage from: {}'.format(args.pseudo_passage_fp))
        else:
            print('not use pseudo passage')
        self.pseudo_psg_data = None
        self.query_to_pseudo_psg_dict = None

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        original_queries = queries
        if args.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
        else:
            input_texts = [self.prompt + q for q in queries]

        q_embeddings = self._do_encode(input_texts)

        if self.query_to_pseudo_psg_dict != None:
            pseudo_psg_list = []
            for q in original_queries:
                pseudo_psg_list.append(self.query_to_pseudo_psg_dict[q])

            pseudo_psg_list = list(map(lambda x:{'text':x},pseudo_psg_list))
            p_embeddings = self.encode_corpus(pseudo_psg_list)

            q_embeddings = (q_embeddings + p_embeddings) / 2

        return q_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if args.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]

        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 24 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, always_add_eos=(args.pool_type == 'last'))
            # print('input_ids:{}'.format(batch_dict['input_ids'].size()))
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0).astype(np.float32)

    def set_prompt(self, prompt: str):
        self.prompt = prompt

    def load_pseudo_passage(self,task_name):
        pseudo_psg_fp = '{}/{}.jsonl'.format(self.args.pseudo_passage_fp,task_name)
        self.pseudo_psg_data = list(jsonlines.open(pseudo_psg_fp))
        query_to_pseudo_psg_dict = {}
        for tmp_js in self.pseudo_psg_data:
            query_to_pseudo_psg_dict[tmp_js['query']] = tmp_js['pseudo_psg']
        self.query_to_pseudo_psg_dict = query_to_pseudo_psg_dict
        logger.info('load pseudo passage from {}'.format(pseudo_psg_fp))


from V6.my_utils import get_task_instruction_by_task_name


class MyRetrievalModel(DRESModel):
    def __init__(self, representation_id, representation_token_num, args):
        if args.pseudo_passage_fp != '0':
            print('use pseudo passage from: {}'.format(args.pseudo_passage_fp))
        else:
            print('not use pseudo passage')
        self.pseudo_psg_data = None
        self.query_to_pseudo_psg_dict = None
        self.task_name = None
        self.see_index = -1
        self.task = None
        self.args = args
        self.query_max_length = self.args.query_max_length
        self.passage_max_length = self.args.passage_max_length
        self.wrap_q_p = args.wrap_q_p

        if args.model_dtype == 'fp16':
            model_dtype = torch.float16
        elif args.model_dtype == 'bf16':
            model_dtype = torch.bfloat16
        elif args.model_dtype == 'fp32':
            model_dtype = torch.float32
        else:
            raise NotImplementedError
        files_in_input_model_dir = os.listdir(args.model_name_or_path)
        if 'adapter_config.json' in files_in_input_model_dir:
            logger.info('the input model is a peft model')
            peft_model_id = args.model_name_or_path
            peft_config = PeftConfig.from_pretrained(peft_model_id)
            if os.path.exists(peft_config.base_model_name_or_path):
                base_model_path = peft_config.base_model_name_or_path
            elif os.path.exists('/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base'):
                base_model_path = '/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base'
            elif os.path.exists('/remote-home/share/models/mistral_7b_base'):
                base_model_path = '/remote-home/share/models/mistral_7b_base'
            else:
                base_model_path = 'mistralai/Mistral-7B-v0.1'
                # raise NotImplementedError
            logger.info('peft use base model at: {}'.format(base_model_path))
            base_model_config = AutoConfig.from_pretrained(base_model_path)
            base_model_config.output_hidden_states = False
            # base_model_config.output_last_hidden_states = True

            base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=model_dtype,
                                                              config=base_model_config,
                                                              # attn_implementation="flash_attention_2",
                                                              )
            model = PeftModel.from_pretrained(base_model, peft_model_id)
            model = model.merge_and_unload()
            self.encoder = model
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path,trust_remote_code=True)
            config.output_hidden_states = False
            logger.info('the input model is a full-finetuned model')
            if 'internlm2' not in args.model_name_or_path:
                self.encoder = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype,
                                                                    config=config,trust_remote_code=True
                                                                    # attn_implementation="flash_attention_2",
                                                                    )
            else:
                self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype,
                                                                    config=config,trust_remote_code=True
                                                                    # attn_implementation="flash_attention_2",
                                                                    )
                if 'ForCausalLM' in str(type(self.encoder)):
                    self.encoder = self.encoder.model

            print('encoder type: {}'.format(type(self.encoder)))
            # if 'internlm2' in args.model_name_or_path:
            #     replace_internlm2_causal_lm_forward_func(self.encoder)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        # self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'right'
        assert self.tokenizer.padding_side == 'right'
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)
        self.encoder.cuda()
        self.encoder.eval()
        self.representation_id = representation_id
        self.representation_token_num = representation_token_num
        self.representation_token = self.tokenizer.convert_ids_to_tokens(self.representation_id)

    def set_task_instruction(self, task_name):
        tmp_instruction = get_task_instruction_by_task_name(task_name)
        self.task_instruction = tmp_instruction
        logger.info('now task instruction: {}'.format(tmp_instruction))

    def wrap_query(self, query):
        if self.wrap_q_p == 'query_or_passage':
            result = 'query: {}'.format(query)
        elif self.wrap_q_p == 'instruction':
            # tmp_instruction = get_task_instruction_by_task_name(self.task_name)
            result = '{}\n{}'.format(self.task_instruction, query)
        else:
            raise NotImplementedError
        return result


    def wrap_passage(self, passage):
        if self.wrap_q_p == 'query_or_passage':
            result = 'passage: {}'.format(passage)
        elif self.wrap_q_p == 'instruction':
            result = passage
        else:
            raise NotImplementedError

        return result


    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        original_queries = queries
        # if args.prefix_type == 'query_or_passage':
        #     input_texts = [f'query: {q}' for q in queries]
        # else:
        #     input_texts = [self.prompt + q for q in queries]

        # input_texts = [add_special_token_for_sequence_embedding(q,)]

        # input_texts = []
        # for q in queries:
        #     input_texts.append(add_special_token_for_sequence_embedding(q, eos_token=self.tokenizer.eos_token,
        #                                                                 representation_token=self.representation_token,
        #                                                                 representation_token_num=self.representation_token_num))

        # input_texts = queries
        # if self.see_index >= 0:
        #     print('original input query:\n{}'.format(input_texts[self.see_index]))


        if self.query_to_pseudo_psg_dict != None:
            if self.args.how_to_use_pseudo_passage == 'embedding_average':
                input_texts = list(map(lambda x: self.wrap_query(x), queries))
                q_embeddings = self._do_encode(input_texts, 'query')
                pseudo_psg_list = []
                for q in original_queries:
                    pseudo_psg_list.append(self.query_to_pseudo_psg_dict[q])

                pseudo_psg_list = list(map(lambda x:{'text':x},pseudo_psg_list))
                p_embeddings = self.encode_corpus(pseudo_psg_list)

                q_embeddings = (q_embeddings + p_embeddings) / 2
            elif self.args.how_to_use_pseudo_passage == 'concat':
                input_texts = []
                for q in queries:
                    pseudo_psg = self.query_to_pseudo_psg_dict[q]
                    q = self.wrap_query(q)
                    query = '{}\nPseudo Passage:\n{}'.format(q, pseudo_psg)
                    input_texts.append(query)

                q_embeddings = self._do_encode(input_texts, 'query')

                # input_texts = list(map(lambda x: self.wrap_query(x), queries))


        else:
            input_texts = list(map(lambda x: self.wrap_query(x), queries))
            q_embeddings = self._do_encode(input_texts, 'query')


        return q_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        # if args.doc_as_query:
        #     return self.encode_queries([d['text'] for d in corpus], **kwargs)
        #
        # input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        # if args.prefix_type == 'query_or_passage':
        #     input_texts = ['passage: {}'.format(t) for t in input_texts]

        corpus = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]

        input_texts = list(map(lambda x: self.wrap_passage(x), corpus))

        # input_texts = corpus

        # input_texts = []
        # for q in corpus:
        #     input_texts.append(add_special_token_for_sequence_embedding(q, eos_token=self.tokenizer.eos_token,
        #                                                                 representation_token=self.representation_token,
        #                                                                 representation_token_num=self.representation_token_num))
        if self.see_index >= 0:
            print('original input passage:\n{}'.format(input_texts[self.see_index]))
        return self._do_encode(input_texts, 'passage')

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], input_type) -> np.ndarray:
        if input_type == 'query':
            max_length = self.query_max_length
        elif input_type == 'passage':
            max_length = self.passage_max_length
        else:
            raise NotImplementedError
        encoded_embeds = []
        batch_size = self.args.batch_per_gpu * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            # batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, always_add_eos=(args.pool_type == 'last'))
            batch_dict = my_create_batch_dict(self.tokenizer, batch_input_texts, self.representation_token,
                                              self.representation_token_num, max_length=max_length)

            # debug start
            # tmp_input_ids = batch_dict['input_ids'][0].cpu()
            # tmp_attn_mask = batch_dict['attention_mask'][0].cpu()
            #
            # tmp_seq_len = torch.sum(tmp_attn_mask)
            # logger.info("input_string:\n{}".format(self.tokenizer.convert_tokens_to_string(
            #     self.tokenizer.convert_ids_to_tokens(tmp_input_ids[:tmp_seq_len]))))
            #
            # logger.info("input_string_pad:\n{}".format(self.tokenizer.convert_tokens_to_string(
            #     self.tokenizer.convert_ids_to_tokens(tmp_input_ids))))
            #
            # if input_type == 'passage':
            #     raise NotImplementedError

            # debug end

            # print(self.tokenizer.convert_ids_to_tokens())
            # raise NotImplementedError
            if self.see_index >= 0 and input_type == 'passage':
                attn_mask = batch_dict['attention_mask'][self.see_index]
                is_not_pad = (attn_mask == 1)
                input_ids = batch_dict['input_ids'][self.see_index][is_not_pad]
                print('input_ids:\n{}'.format(input_ids))
                print()
                print(
                    'input_tokens:\n{}'.format(self.tokenizer.convert_ids_to_tokens(input_ids)))
                print()
                print('input_string:\n{}'.format(self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids))))
                print()

                exit()

            batch_dict = move_to_cuda(batch_dict)
            # print('batch_dict:{}'.format(batch_dict.keys()))
            # logger.info('input_ids:{}'.format(batch_dict['input_ids'].size()))
            batch_dict['output_hidden_states']=False
            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)

                input_ids = batch_dict['input_ids']
                attention_mask = batch_dict['attention_mask']
                # print('input_ids:{}'.format(input_ids.size()))
                tmp_batch_size = input_ids.size()[0]
                seq_len = input_ids.size()[1]

                is_representation_id = (input_ids == self.representation_id)

                range_tensor = torch.arange(seq_len).unsqueeze(0).to(is_representation_id.device)
                # eos_id_pos = torch.where(input_ids == self.tokenizer.eos_token_id)[1]
                seq_len = torch.sum(attention_mask, dim=1)

                first_representation_token_pos = seq_len - (self.representation_token_num)

                # last_representation_token_pos_1 = seq_len

                mask = range_tensor < (first_representation_token_pos.unsqueeze(1))
                # mask the representation_token in the original input
                is_representation_id[mask] = False

                # encoded = self.model(*args, **kwargs)
                # outputs
                # last_hidden_states = outputs.hidden_states[-1]
                last_hidden_states = outputs.last_hidden_state
                hidden_size = last_hidden_states.size()[-1]

                # print('is_representation_id:\n{}'.format(is_representation_id))
                # print('is_representation_id.sum:{}'.format(torch.sum(is_representation_id, dim=1)))

                sequence_representation_embeds = last_hidden_states[is_representation_id]
                sequence_representation_embeds = sequence_representation_embeds.view(tmp_batch_size, -1, hidden_size)
                sequence_representation = torch.mean(sequence_representation_embeds, dim=1)
                # sequence_representation = sequence_representation / torch.norm(sequence_representation, p=2, dim=1,
                #                                                                keepdim=True)
                sequence_representation = nn.functional.normalize(sequence_representation, p=2, dim=-1)

                # print('sequence_representation:{}'.format(sequence_representation.size()))

                encoded_embeds.append(sequence_representation.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0).astype(np.float32)

    # def set_prompt(self, prompt: str):
    #     self.prompt = prompt
    def load_pseudo_passage(self,task_name):
        pseudo_psg_fp = '{}/{}.jsonl'.format(self.args.pseudo_passage_fp,task_name)
        self.pseudo_psg_data = list(jsonlines.open(pseudo_psg_fp))
        query_to_pseudo_psg_dict = {}
        for tmp_js in self.pseudo_psg_data:
            query_to_pseudo_psg_dict[tmp_js['query']] = tmp_js['pseudo_psg']
        self.query_to_pseudo_psg_dict = query_to_pseudo_psg_dict
        logger.info('load pseudo passage from {}'.format(pseudo_psg_fp))


def main():
    assert AbsTaskRetrieval.is_dres_compatible(RetrievalModel)
    assert AbsTaskRetrieval.is_dres_compatible(MyRetrievalModel)
    assert AbsTaskRetrieval.is_dres_compatible(RepLLAMARetrievalModel)

    if args.force_retrieval_model != 'default':
        if args.force_retrieval_model == 'repllama':
            print('retrieval model choose RepLLAMARetrievalModel')
            model = RepLLAMARetrievalModel()
        elif args.force_retrieval_model == 'original':
            print('retrieval model choose RetrievalModel')
            model = RetrievalModel()
        elif args.force_retrieval_model == 'mine':
            print('retrieval model choose RetrievalModel')
            model = MyRetrievalModel(args.representation_id, args.representation_token_num, args)
        else:
            raise NotImplementedError("args.force_retrieval_model:{}".format(args.force_retrieval_model))

    elif 'repllama' in args.model_name_or_path:
        print('retrieval model choose RepLLAMARetrievalModel')
        model = RepLLAMARetrievalModel()
    elif args.model_name_or_path in ['intfloat/e5-mistral-7b-instruct'] or 'e5_mistral' in args.model_name_or_path:
        print('retrieval model choose RetrievalModel')
        model = RetrievalModel()
    # elif 'BAAI/bge' in args.model_name_or_path:
    #     print('retrieval model choose bge')
        # model = BGERetrievalModel(args)
    else:
        print('retrieval model choose MyRetrievalModel')
        model = MyRetrievalModel(args.representation_id, args.representation_token_num, args)

    logger.info('retrieval model type: {}'.format(type(model)))

    task_names = [t.description["name"] for t in MTEB(task_types=['Retrieval'], task_langs=['en']).tasks]
    task_names = [t for t in task_names if t != 'MSMARCOv2']
    logger.info('Tasks: {}'.format(task_names))

    # print(task_names)
    # ['ArguAna', 'ClimateFEVER', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval',
    #  'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval',
    #  'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval',
    #  'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval',
    #  'CQADupstackWordpressRetrieval', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ',
    #  'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID']
    #
    # exit()
    # 'ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA MSMARCO NFCorpus NQ QuoraRetrieval SCIDOCS SciFact Touche2020 TRECCOVID'
    # 'ArguAna ClimateFEVER DBPedia FEVER FiQA2018 HotpotQA MSMARCO NFCorpus NQ QuoraRetrieval SCIDOCS SciFact Touche2020 TRECCOVID'

    if args.task_names == None or len(args.task_names) == 0:
        task_names = ['ArguAna', 'FiQA2018', 'NFCorpus', 'SciFact', 'SCIDOCS', 'QuoraRetrieval', 'TRECCOVID',
                      'Touche2020']
    else:
        task_names = args.task_names.split(' ')

    for task in task_names:
        # if task in ['ArguAna']:
        #     continue
        if args.dry_run and task not in ['SciFact', 'FiQA2018']:
            continue

        logger.info('Processing task: {}'.format(task))

        if type(model) != MyRetrievalModel and args.prefix_type == 'query_or_passage':
            args.doc_as_query = task in ['QuoraRetrieval']
        else:
            task_def: str = get_task_def_by_task_name_and_type(task_name=task, task_type='Retrieval')
            prompt: str = get_detailed_instruct(task_def)
            if type(model) in [RetrievalModel, RepLLAMARetrievalModel]:
                model.set_prompt(prompt=prompt)
                logger.info('Set prompt: {}'.format(prompt))
            elif type(model) == MyRetrievalModel:
                model.set_task_instruction(task)
            else:
                raise NotImplementedError

        if args.pseudo_passage_fp != '0':
            model.load_pseudo_passage(task)
        else:
            logger.info('do not use pseudo passage')
        if args.doc_embedding_cache_dir != None:
            doc_embedding_cache_dir_this_task = '{}/{}'.format(args.doc_embedding_cache_dir, task)
            os.makedirs(doc_embedding_cache_dir_this_task, exist_ok=True)
        else:
            doc_embedding_cache_dir_this_task = None
        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, eval_splits=["test" if task not in ['MSMARCO'] else 'dev'],
                       output_folder=args.output_dir, doc_embedding_cache_dir=doc_embedding_cache_dir_this_task)


if __name__ == '__main__':
    main()
