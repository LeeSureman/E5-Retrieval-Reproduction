import os.path

from transformers import AutoModel
import argparse
import jsonlines
from fastchat.model import get_conversation_template
import tqdm
from utils import _setup_logger
logger = _setup_logger()
from vllm import LLM, SamplingParams
import torch

# import accelerate



hyde_prompt_template_dict = {
    'ArguAna': 'Please write a counter argument for the passage.\nPassage: {}\nCounter Argument:',
    'FiQA2018': 'Please write a financial article passage to answer the question.\nQuestion: {}\nFinancial Article Passage:',
    'MSMARCO': 'Please write a passage to answer the question.\nQuestion: {}\nPassage:',
    'NFCorpus': 'Please write a medical paper passage to answer the question or explain the phrase.\nQuestion/Phrase: {}\nMedical Paper Passage:',
    'NQ': 'Please write a passage to answer the question.\nQuestion: {}\nPassage:',
    'SCIDOCS': 'Please write a scientific paper passage for the scientific paper title.\nScientific Paper Title: {}\nScientific Paper Passage:',
    'SciFact': 'Please write a scientific paper passage to support or refute the claim.\nClaim: {}\nScientific Paper Passage:',
    'TRECCOVID': 'Please write a scientific paper passage to answer the question.\nQuestion: {}\nScientific Paper Passage:',
    'Touche2020': 'Please write a detailed and persuasive argument for the question.\nQuestion: {}\nArgument:',
    'FEVER': 'Please write a passage to support/refute the claim.\nClaim: {}\nPassage:',
    'HotpotQA': 'Please write a passage to answer the multi-hop question.\nQuestion: {}\nPassage:'
}
hyde_prompt_template_dict['nq'] = hyde_prompt_template_dict['NQ']
hyde_prompt_template_dict['fever'] = hyde_prompt_template_dict['FEVER']
hyde_prompt_template_dict['msmarco_passage'] = hyde_prompt_template_dict['MSMARCO']
hyde_prompt_template_dict['hotpot_qa'] = hyde_prompt_template_dict['HotpotQA']

hyde_prompt_template_candidate_dict = {}
hyde_prompt_template_candidate_dict['HotpotQA'] = [
    'Please write a passage to answer the question.\nQuestion: {}\nPassage:',
    'Please write a detailed passage to answer the multi-hop question.\nQuestion: {}\nPassage:',
    'Please write a relevant passage to answer the multi-hop question.\nQuestion: {}\nPassage:',
    'Given the following multi-hop question, please write a passage to answer it.\nQuestion: {}\nPassage:',
    'Please write a document to answer the question.\nQuestion: {}\nDocument:',
]

hyde_prompt_template_candidate_dict['FEVER'] = [
    'Please write a detailed passage to support/refute the claim.\nClaim: {}\nPassage:',
    'Please write a passage to explain the factual correctness of the claim.\nClaim: {}\nPassage:',
    'Please write a passage to explain the factuality of the claim.\nClaim: {}\nPassage:',
    'Given the following claim, please write a passage to support/refute it.\nClaim: {}\nPassage:',
    'Please write a document to support/refute the claim.\nClaim: {}\nDocument:',

]

hyde_prompt_template_candidate_dict['MSMARCO'] = [
    'Please write a passage to answer the web search query.\nQuery: {}\nPassage:',
    'Please write a passage for the web search query.\nQuery: {}\nPassage:',
    'Please write a document for the web search query.\nQuery: {}\nDocument:',
    'Please write a detailed passage to answer the web search query.\nQuery: {}\nPassage:',
    'Please write a relevant passage to answer the web search query.\nQuery: {}\nPassage:',
    'Given the following web search query, please write a passage for it.\nQuery: {}\nPassage:',

]

hyde_prompt_template_candidate_dict['NQ'] = [
    'Please write a detailed passage to answer the question.\nQuestion: {}\nPassage:',
    'Please write a relevant passage to answer the question.\nQuestion: {}\nPassage:',
    'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:',
    'Please write a document to answer the question.\nQuestion: {}\nDocument:',
]


def wrap_user_input_for_inference(user_input, sft_template_type):
    if sft_template_type == 'vicuna':
        conv_vicuna = get_conversation_template('vicuna')
        conv_vicuna.append_message(conv_vicuna.roles[0], user_input)
        result = conv_vicuna.get_prompt() + conv_vicuna.roles[1] + ':'
        return result
    elif sft_template_type == 'mistral':
        conv_mistral = get_conversation_template('mistral')
        conv_mistral.append_message(conv_mistral.roles[0], user_input)
        result = conv_mistral.get_prompt() + conv_mistral.roles[1]
        return result
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--query_dir')
    parser.add_argument('--tasks')
    parser.add_argument('--temperature',type=float)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--retrieval_query_max_length',type=int,required=True)
    parser.add_argument('--model_dtype',required=True)
    parser.add_argument('--sft_template',required=True,choices=['mistral','vicuna'])
    parser.add_argument('--use_prompt_candidate_template',default=-10000,type=int)
    parser.add_argument('--sample_n',type=int,default=1)
    parser.add_argument('--max_num_seqs',type=int)
    parser.add_argument('--total_gpu',type=int,required=True)

    args = parser.parse_args()
    args.tasks = args.tasks.split()



    # args.use_prompt_candidate_template = args.args.use_prompt_candidate_template.split()
    # use_prompt_candidate_template = {}
    # use_prompt_candidate_template_task = args.use_prompt_candidate_template

    #VLLM related

    # if torch.cuda.device_count()==8:
    #     tensor_parallel_size = 8
    # elif torch.cuda.device_count() >=4:
    #     tensor_parallel_size = 4
    # elif torch.cuda.device_count() >=2:
    #     tensor_parallel_size = 2
    # else:
    #     tensor_parallel_size = 1


        # if dist.get_rank() == 0:
        #     print('wait for other process finish, then merge the outputs')
        # dist.barrier()
    for task in args.tasks:
        print('start merge task {} files'.format(task))
        query_fp = '{}/{}.jsonl'.format(args.query_dir, task)
        query_js_s = list(jsonlines.open(query_fp))
        queries = list(map(lambda x: x['query'], query_js_s))

        qid_and_query_s = list(enumerate(queries))
        all_qid_and_query_s = qid_and_query_s

        all_output_js_s = []

        for i in range(args.total_gpu):
            tmp_output_fp = '{}/{}/shard_{}.jsonl'.format(args.output_dir, task, i)
            tmp_output_js_s = list(jsonlines.open(tmp_output_fp))
            all_output_js_s.extend(tmp_output_js_s)

        all_output_js_s.sort(key=lambda x:x['qid'])

        original_all_qids = list(map(lambda x:x[0],all_qid_and_query_s))
        qids_in_all_outputs = list(map(lambda x:x['qid'],all_output_js_s))

        assert original_all_qids == qids_in_all_outputs

        entire_output_fp = '{}/{}.jsonl'.format(args.output_dir, task)
        with jsonlines.open(entire_output_fp, 'w') as f_out:
            f_out.write_all(all_output_js_s)

        #check qids

        logger.info('finish task {}'.format(task))



