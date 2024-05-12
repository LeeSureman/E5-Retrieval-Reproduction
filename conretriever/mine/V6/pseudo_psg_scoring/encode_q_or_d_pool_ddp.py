import copy
import os

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import transformers
import torch
import jsonlines
import tqdm
import argparse
import sys
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np

sys.path.append('..')
sys.path.append('.')
import logging
from multiprocessing import Pool

from mine.V6 import prompt_config


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def add_special_token_for_sequence_embedding(input_str, eos_token, representation_token, representation_token_num):
    result = representation_token * representation_token_num
    return result


def tokenize_into_ids_with_special_token_for_sequence_embedding(input_str, tokenizer, representation_token,
                                                                representation_token_num, max_length):
    assert tokenizer.add_bos_token
    representation_special_tokens = add_special_token_for_sequence_embedding('', tokenizer.eos_token,
                                                                             representation_token=representation_token,
                                                                             representation_token_num=representation_token_num)
    representation_special_ids = tokenizer(representation_special_tokens)['input_ids'][1:]

    representation_special_tokens_num = len(representation_special_ids)

    input_str_encoded = tokenizer(text=input_str,
                                  max_length=max_length - representation_special_tokens_num,
                                  return_token_type_ids=False,
                                  return_attention_mask=False,
                                  padding=False,
                                  truncation=True
                                  )

    input_str_encoded['input_ids'] = input_str_encoded['input_ids'] + representation_special_ids

    result = tokenizer.pad(
        input_str_encoded,
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors='pt',
        max_length=max_length
    )

    return result


class My_Dataset(Dataset):
    def __init__(self, id_with_inputs, tokenizer, representation_token, representation_token_num, max_length):
        self.id_with_inputs = id_with_inputs
        self.tokenizer = tokenizer
        self.representation_token = representation_token
        self.representation_token_num = representation_token_num
        self.max_length = max_length
        assert len(id_with_inputs[0]) == 2 and type(id_with_inputs[0]) == type([])

        self.is_tokenized = False

    def __getitem__(self, i):
        # result = self.tokenizer(self.data[i],
        #                         max_length=1024,
        #                         padding=True,
        #                         return_token_type_ids=False,
        #                         return_attention_mask=True,
        #                         truncation=True,
        #                         return_tensors="pt"
        #                         )
        # result['input_ids'] = result['input_ids'][0]
        if not self.is_tokenized:
            result = tokenize_into_ids_with_special_token_for_sequence_embedding(
                self.id_with_inputs[i][1], self.tokenizer, self.representation_token, self.representation_token_num,
                self.max_length)
            self.id_with_inputs[i][1] = result
        else:
            result = self.id_with_inputs[i][1]

        return result

    def __len__(self):
        return len(self.id_with_inputs)


def wrap_query_with_prefix(query, wrap_q_p, task=None, instruction=None):
    instruction_template = 'Given the following {}, give me {}.'
    task_to_query_type = prompt_config.task_to_query_type
    task_to_passage_type = prompt_config.task_to_passage_type

    if wrap_q_p == 'query_or_passage':
        assert NotImplementedError('only instruction prefix for main experiments')
        result = 'query: {}'.format(query)
    elif wrap_q_p == 'instruction':
        assert task != None
        if task != 'synthetic':
            assert instruction == None
            tmp_instruction = instruction_template.format(task_to_query_type[task],
                                                          task_to_passage_type[task])
        else:
            assert instruction != None
            tmp_instruction = instruction.strip()
        result = '{}\n{}'.format(tmp_instruction, query)
    else:
        raise NotImplementedError
    return result


def wrap_query_with_pseudo_psg(query, pseudo_psg):
    result = '{}\nPseudo Passage:\n{}'.format(query, pseudo_psg)
    return result


def wrap_doc(passage):
    # 和之前对psg的处理保持一致，把分隔title和content的双换行替换为空格
    passage_split = passage.split('\n\n')
    title = passage_split[0]
    content = '\n\n'.join(passage_split[1:])
    passage = '{} {}'.format(title, content).strip()

    return passage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    # parser.add_argument('--q_to_pos_dir',required=True)
    parser.add_argument('--input_fp', required=True)
    parser.add_argument('--encode_query_or_doc', choices=['query', 'doc'])
    parser.add_argument('--representation_id', type=int, required=True)
    parser.add_argument('--representation_token_num', type=int, required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--use_flash_attention', type=int, required=True)
    # parser.add_argument('--tokenize_process',type=int,required=True)

    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--shard_index', type=int, required=True)
    parser.add_argument('--total_gpu', type=int, required=True)
    parser.add_argument('--sort_length_when_encoding', type=int, required=True)
    parser.add_argument('--pre_tokenize_all_input',type=int,required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.unk_token

    if args.use_flash_attention:
        tokenizer.padding_side = 'left'
    else:
        assert tokenizer.padding_side == 'right'

    representation_token = tokenizer.convert_ids_to_tokens(args.representation_id)

    logger.info('load data at {} start'.format(args.input_fp))
    input_data = list(jsonlines.open(args.input_fp))
    logger.info('load data at {} finish'.format(args.input_fp))

    # input_data = input_data[:200]

    if args.encode_query_or_doc == 'doc':
        assert 'id' in input_data[0] and 'contents' in input_data[0]
    elif args.encode_query_or_doc == 'query':
        assert 'query' in input_data[0] and 'pseudo_psg' in input_data[0]
        assert type(input_data[0]['pseudo_psg']) == type([])
        pseudo_psg_sample_n = len(input_data[0]['pseudo_psg'])
        for i, tmp_js in enumerate(tqdm.tqdm(input_data, desc='check pseudo psg each num')):
            assert len(tmp_js['pseudo_psg']) == pseudo_psg_sample_n, i

    data_collator = transformers.DataCollatorWithPadding(tokenizer, max_length=args.max_length, padding='longest')

    global_idx_with_input_data = list(enumerate(input_data))

    gidx_with_input_data_this_shard = list(
        filter(lambda x: x[0] % args.total_gpu == args.shard_index, global_idx_with_input_data))

    del input_data



    task_name = args.input_fp.split('/')[-1].split('.')[0]
    logger.info('task_name: {}'.format(task_name))

    if args.encode_query_or_doc == 'doc':

        doc_inputs = []
        logger.info('start pre tokenize data')
        for gidx, doc_js in tqdm.tqdm(gidx_with_input_data_this_shard):
            doc = wrap_doc(doc_js['contents'])
            # doc_input = tokenize_into_ids_with_special_token_for_sequence_embedding(
            #     doc, tokenizer, representation_token, args.representation_token_num, args.max_length)
            doc_inputs.append([gidx, doc])
        # logger.info('finish pre tokenize data')

        my_dataset = My_Dataset(doc_inputs,
                                tokenizer, representation_token, args.representation_token_num, args.max_length)

    elif args.encode_query_or_doc == 'query':
        query_inputs = []
        for gidx, tmp_js in tqdm.tqdm(gidx_with_input_data_this_shard, disable=1):
            query = tmp_js['query']
            if 'instruction' not in tmp_js:
                query = wrap_query_with_prefix(query, 'instruction', task=task_name)
            else:
                assert task_name == 'synthetic'
                query = wrap_query_with_prefix(query, 'instruction', task=task_name, instruction=tmp_js['instruction'])

            query_without_ppsg = query
            # query_input = tokenize_into_ids_with_special_token_for_sequence_embedding(
            #     query_without_ppsg, tokenizer, representation_token, args.representation_token_num, args.max_length)
            query_inputs.append([gidx*100+pseudo_psg_sample_n, query_without_ppsg])

            for j in range(pseudo_psg_sample_n):
                pseudo_psg = tmp_js['pseudo_psg'][j]
                query_with_ppsg = wrap_query_with_pseudo_psg(query, pseudo_psg)

                # query_input = tokenize_into_ids_with_special_token_for_sequence_embedding(
                # query_with_ppsg, tokenizer, representation_token, args.representation_token_num, args.max_length)
                query_inputs.append([gidx*100+j, query_with_ppsg])

        assert len(query_inputs) % (pseudo_psg_sample_n + 1) == 0
        my_dataset = My_Dataset(query_inputs,
                                tokenizer, representation_token, args.representation_token_num, args.max_length)

    print('my_dataset size: {}'.format(len(my_dataset)))

    if args.pre_tokenize_all_input:
        pass
        pre_tokenize_cache_fp = '{}_tokenized_cache/shard_{}.pkl'.format(args.output_dir, args.shard_index)
        if os.path.exists(pre_tokenize_cache_fp):
            my_dataset.id_with_inputs = torch.load(open(pre_tokenize_cache_fp, 'rb'))
        else:
            for i, tmp_js in enumerate(tqdm.tqdm(my_dataset,disable=args.shard_index!=0)):
                pass

            os.makedirs('{}_tokenized_cache'.format(args.output_dir),exist_ok=True)
            torch.save(my_dataset.id_with_inputs,open(pre_tokenize_cache_fp, 'wb'))
        my_dataset.is_tokenized = True

        # for pre-tokenize:
        # pseudo_data_loader = torch.utils.data.DataLoader(
        #     my_dataset, shuffle=False, batch_size=1, num_workers=4, collate_fn=data_collator, prefetch_factor=2)
        #
        # tokenized_inputs = []
        # for tmp_batch in tqdm.tqdm(pseudo_data_loader):
        #     tmp_batch_size = tmp_batch['input_ids'].size()[0]
        #     assert tmp_batch_size == 1
        #     for k in tmp_batch:
        #         tmp_batch[k] = tmp_batch[k][0]
        #     tokenized_inputs.append(tmp_batch)
        #     # print(t)
        #     # tokenized_inputs.extend(tmp_batch)
        # my_dataset.is_tokenized = True
        #
        # for i, tmp in enumerate(my_dataset.id_with_inputs):
        #     tmp[1] = tokenized_inputs[i]
        #
        # del pseudo_data_loader





    if args.sort_length_when_encoding:
        if my_dataset.is_tokenized:
            print('sort input order by input_ids num to be faster')
            my_dataset.id_with_inputs.sort(key=lambda x: len(x[1]['input_ids']), reverse=True)
        else:
            print('sort input order by word num to be faster')
            my_dataset.id_with_inputs.sort(key=lambda x: len(x[1].split()), reverse=True)
    else:
        print('not sort input order by length to be faster')

    # print('my_dataset.id_with_inputs[0][1].keys(): {}'.format(my_dataset.id_with_inputs[0][1].keys()))

    print('inputs keys: {}'.format(my_dataset[0].keys()))
    for k, v in my_dataset[0].items():
        print('{}: {}'.format(k, v.size()))
    print('my_dataset[0]: ', my_dataset[0]['input_ids'].size())
    print('my_dataset[-1]: ', my_dataset[-1]['input_ids'].size(), my_dataset[-1])



    # for pre-tokenize:
    # pseudo_data_loader =

    my_data_loader = torch.utils.data.DataLoader(
        my_dataset, shuffle=False, batch_size=args.batch_size, num_workers=16, collate_fn=data_collator)

    device = torch.device('cuda')
    if args.use_flash_attention:
        logger.info('model use flash attention')
        model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16,
                                          attn_implementation="flash_attention_2", )
    else:
        logger.info('model not use flash attention')
        model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)

    print('model type: {}'.format(type(model)))
    model.eval()
    model = model.to(device)

    encoded_embeds = []
    with torch.no_grad():
        for batch in tqdm.tqdm(my_data_loader,disable=args.shard_index!=0):
            batch = batch.to(device)

            outputs = model(**batch)

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            tmp_batch_size = input_ids.size()[0]
            seq_len = input_ids.size()[1]

            is_representation_id = (input_ids == args.representation_id)

            range_tensor = torch.arange(seq_len).unsqueeze(0).to(is_representation_id.device)


            seq_len = torch.sum(attention_mask, dim=1)

            first_representation_token_pos = seq_len - (args.representation_token_num)



            mask = range_tensor < (first_representation_token_pos.unsqueeze(1))
            # mask the representation_token in the original input
            is_representation_id[mask] = False


            last_hidden_states = outputs.last_hidden_state
            hidden_size = last_hidden_states.size()[-1]



            sequence_representation_embeds = last_hidden_states[is_representation_id]
            sequence_representation_embeds = sequence_representation_embeds.view(tmp_batch_size, -1, hidden_size)
            sequence_representation_embeds = sequence_representation_embeds.to(torch.float32)
            sequence_representation = torch.mean(sequence_representation_embeds, dim=1)

            sequence_representation = nn.functional.normalize(sequence_representation, p=2, dim=-1)

            encoded_embeds.append(sequence_representation.cpu())

            # print('encoded_embeds dtype: {}'.format(encoded_embeds[0].dtype))

    encoded_embeds_all = torch.cat(encoded_embeds, dim=0).to(torch.float16)
    # ( len(my_dataset.id_with_inputs), hidden_size )

    # id here is gidx for "doc" and gidx*100 + pseudo_psg_idx (the last is no psg) for "query"
    assert len(my_dataset.id_with_inputs) == len(encoded_embeds_all)

    # now my_dataset.id_with_inputs, gidx_s and encoded_embeds_all is all sorted by input length (reverse)
    gidxs = list(map(lambda x:x[0], my_dataset.id_with_inputs))
    assert len(gidxs) == len(encoded_embeds_all), "{}, {}".format(len(gidxs), len(encoded_embeds_all))
    gidxs_with_embeds = list(zip(gidxs, encoded_embeds_all))

    sorted_gidxs_with_embeds = sorted(gidxs_with_embeds, key=lambda x:x[0])


    output_js_s = []

    if args.encode_query_or_doc == 'doc':
        assert len(gidx_with_input_data_this_shard) == len(sorted_gidxs_with_embeds)
        for (gidx_1, input_js), (gidx_2, embed) in list(zip(gidx_with_input_data_this_shard, sorted_gidxs_with_embeds)):
            assert gidx_1 == gidx_2
            output_js = {}
            output_js['gidx'] = gidx_1
            output_js['id'] = input_js['id']
            output_js['contents'] = input_js['contents']
            output_js['embed'] = embed

            output_js_s.append(output_js)
    elif args.encode_query_or_doc == 'query':
        assert len(gidx_with_input_data_this_shard) * (pseudo_psg_sample_n+1) == len(sorted_gidxs_with_embeds)

        for i, (gidx, input_js) in enumerate(gidx_with_input_data_this_shard):
            tmp_embeds = []
            tmp_start = i*(pseudo_psg_sample_n+1)
            tmp_end = (i+1)*(pseudo_psg_sample_n+1)
            #"+1" corresponds to "without pseudo_psg"
            subgidxs_with_embeds = sorted_gidxs_with_embeds[tmp_start: tmp_end]
            for j, (subgidx, embed) in enumerate(subgidxs_with_embeds):
                assert subgidx == 100*gidx + j, "{} == {}".format(subgidx, 100*gidx + j)
                tmp_embeds.append(embed)
            output_js = copy.copy(input_js)
            output_js['gidx'] = gidx
            output_js['embeds'] = tmp_embeds
            output_js_s.append(output_js)

    output_fp = '{}/shard_{}.pkl'.format(args.output_dir, args.shard_index)
    with open(output_fp, 'wb') as f_out:
        print('start writing files in {}'.format(output_fp))
        torch.save(output_js_s, f_out)
        print('finish writing files in {}'.format(output_fp))





