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
    parser.add_argument('--shard_index',type=int,required=True)
    parser.add_argument('--total_gpu',type=int,required=True)

    args = parser.parse_args()
    args.tasks = args.tasks.split()

    if args.shard_index >= args.total_gpu:
        print('args.shard_index {} >= args.total_gpu {} , so exit!'.format(args.shard_index, args.total_gpu))
        exit()

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
    tensor_parallel_size=1

    logger.info('use tensor_parallel_size:{}'.format(tensor_parallel_size))
    logger.info('global num gpu: {}'.format(args.total_gpu))
    if args.sample_n <=1:
        args.sample_n = 1
    sampling_params = SamplingParams(n=args.sample_n, temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=tensor_parallel_size,
              max_model_len=1024,dtype=args.model_dtype, gpu_memory_utilization=0.9,
              max_num_seqs=args.max_num_seqs)




    for task in args.tasks:
        os.makedirs('{}/{}'.format(args.output_dir, task),exist_ok=True)
        output_fp = '{}/{}/shard_{}.jsonl'.format(args.output_dir, task, args.shard_index)
        if os.path.exists(output_fp):
            logger.info('{} already exsits, so continue'.format(output_fp))
            continue
        logger.info('start task {}'.format(task))
        query_fp = '{}/{}.jsonl'.format(args.query_dir, task)
        query_js_s = list(jsonlines.open(query_fp))
        queries = list(map(lambda x: x['query'], query_js_s))

        qid_and_query_s = list(enumerate(queries))

        qid_and_query_s_this_shard = list(filter(lambda x:x[0] % args.total_gpu == args.shard_index, qid_and_query_s))

        # qids =
        all_qid_and_query_s = qid_and_query_s
        qid_and_query_s = qid_and_query_s_this_shard



        prompts = []
        if args.use_prompt_candidate_template >=0:

            hyde_prompt_template = hyde_prompt_template_candidate_dict[task][args.use_prompt_candidate_template]
            logger.info('task {} use candidate prompt template: {}'.format(task, hyde_prompt_template))
        else:
            hyde_prompt_template = hyde_prompt_template_dict[task]
        print('*'*40)
        print('hyde prompt example:')
        print(hyde_prompt_template.format(queries[0]))
        print('*' * 40)
        for qid, q in qid_and_query_s:
            q = ' '.join(q.split(' ')[:args.retrieval_query_max_length])
            hyde_prompt = hyde_prompt_template.format(q)
            final_prompt = wrap_user_input_for_inference(hyde_prompt, args.sft_template)
            prompts.append(final_prompt)

        outputs_vllm = llm.generate(prompts, sampling_params, use_tqdm=(args.shard_index==0))

        pseudo_passage_s = []
        if args.sample_n <=1:
            for output in outputs_vllm:
                pseudo_passage_s.append(output.outputs[0].text.strip())
        else:
            for output in outputs_vllm:
                pseudo_passage_s.append(list(map(lambda x:x.text.strip(),output.outputs)))

        output_js_s = []
        for i, (qid, query) in enumerate(qid_and_query_s):
            output_js = {'query': query, 'pseudo_psg': pseudo_passage_s[i], 'qid':qid}
            output_js_s.append(output_js)

        with jsonlines.open(output_fp,'w') as f_out:
            for tmp_js in output_js_s:
                f_out.write(tmp_js)

        print('finish shard {} task {}'.format(args.shard_index, task))

    print('finish shard {} all task'.format(args.shard_index))

        # if dist.get_rank() == 0:
        #     print('wait for other process finish, then merge the outputs')
        # dist.barrier()
        #
        # all_output_js_s = []
        #
        # for i in range(dist.get_world_size()):
        #     tmp_output_fp = '{}/{}/shard_{}.jsonl'.format(args.output_dir, task, i)
        #     tmp_output_js_s = list(jsonlines.open(tmp_output_fp))
        #     all_output_js_s.extend(tmp_output_js_s)
        #
        # all_output_js_s.sort(key=lambda x:x['qid'])
        #
        # original_all_qids = list(map(lambda x:x[0],all_qid_and_query_s))
        # qids_in_all_outputs = list(map(lambda x:x['qid'],all_qid_and_query_s))
        #
        # assert original_all_qids == qids_in_all_outputs
        #
        # entire_output_fp = '{}/{}.jsonl'.format(args.output_dir, task)
        # with jsonlines.open(entire_output_fp, 'w') as f_out:
        #     f_out.write_all(all_output_js_s)
        #
        # #check qids
        #
        # logger.info('finish task {}'.format(task))



