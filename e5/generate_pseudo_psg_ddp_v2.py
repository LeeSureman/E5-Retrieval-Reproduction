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
    # 'ArguAna': 'Please write a counter argument for the passage.\nPassage: {}\nCounter Argument:',
    'ArguAna': 'Given the following argument, please write a passage to argue against it.\nArgument: {}\nPassage:',
    # 'FiQA2018': 'Please write a financial article passage to answer the question.\nQuestion: {}\nFinancial Article Passage:',
    # 'NFCorpus': 'Please write a medical paper passage to answer the question or explain the phrase.\nQuestion/Phrase: {}\nMedical Paper Passage:',
    'NFCorpus': 'Given the following medical title, please write a medical article passage for it.\nTitle: {}\nPassage:',
    # 'SCIDOCS': 'Please write a scientific paper passage for the scientific paper title.\nScientific Paper Title: {}\nScientific Paper Passage:',
    'SCIDOCS': 'Given the following scientific paper tile, please write relevant research abstracts for it.\nTitle: {}\nPassage:',
    # 'SciFact': 'Please write a scientific paper passage to support or refute the claim.\nClaim: {}\nScientific Paper Passage:',
    'SciFact': 'Given the following claim, please write a scientific paper passage to support/refute it.\nClaim: {}\nPassage:',
    # 'TRECCOVID': 'Please write a scientific paper passage to answer the question.\nQuestion: {}\nScientific Paper Passage:',
    'TRECCOVID': 'Given the following question on COVID-19, please write a scientific paper passage to answer it.\nQuestion: {}\nPassage',
    # 'Touche2020': 'Please write a detailed and persuasive argument for the question.\nQuestion: {}\nArgument:',
    'Touche2020': 'Given the following question, please write a detailed and persuasive argument passage for it.\nQuestion: {}\nPassage',
    # 'HotpotQA': 'Please write a passage to answer the multi-hop question.\nQuestion: {}\nPassage:',
    # 'FEVER': 'Please write a passage to support/refute the claim.\nClaim: {}\nPassage:',
    # 'NQ': 'Please write a passage to answer the question.\nQuestion: {}\nPassage:',
    # 'MSMARCO': 'Please write a passage to answer the question.\nQuestion: {}\nPassage:',

    'NQ': 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:',
    'FEVER': 'Given the following claim, please write a passage to support/refute it.\nClaim: {}\nPassage:',
    'ClimateFEVER': 'Given the following claim, please write a passage to support/refute it.\nClaim: {}\nPassage:',
    'HotpotQA': 'Given the following multi-hop question, please write a passage to answer it.\nQuestion: {}\nPassage:',
    'MSMARCO': 'Given the following web search query, please write a passage for it.\nQuery: {}\nPassage:',
    'FiQA2018': 'Given a financial question, please write a financial article passage to answer it.\nQuestion: {}\nPassage:',
    'DBPedia': 'Given a query, please write a relevant passage for it.\nQuery: {}\nPassage:'
}
hyde_prompt_template_dict['nq'] = hyde_prompt_template_dict['NQ']
hyde_prompt_template_dict['fever'] = hyde_prompt_template_dict['FEVER']
hyde_prompt_template_dict['msmarco_passage'] = hyde_prompt_template_dict['MSMARCO']
hyde_prompt_template_dict['hotpot_qa'] = hyde_prompt_template_dict['HotpotQA']
hyde_prompt_template_dict['fiqa'] = hyde_prompt_template_dict['FiQA2018']

hyde_prompt_template_dict['eli5'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
hyde_prompt_template_dict['squad'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
hyde_prompt_template_dict['triviaqa'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
hyde_prompt_template_dict['quora_duplicate'] = 'Given the following question, please write 10 diverse paraphrased versions of it.\nQuestion: {}\nParaphrases:'
hyde_prompt_template_dict['QuoraRetrieval'] = hyde_prompt_template_dict['quora_duplicate']

hyde_prompt_template_dict['agnews'] = 'Given the following news title, please write a corresponding news article.\nNews Title: {}\nNews Articles'
hyde_prompt_template_dict['amazon_qa'] = 'Given the following Amazon user question, please write a detailed answer for it.\nQuestion: {}\nAnswer:'
hyde_prompt_template_dict['amazon_review'] = 'Given the following Amazon review title, please write a corresponding review.\nReview Title: {}\nReview:'
hyde_prompt_template_dict['ccnews_title_text'] = 'Given the following news title, please write a corresponding news article.\nNews Title: {}\nNews Articles'
hyde_prompt_template_dict['npr'] = 'Given the following news title, please write a corresponding news article.\nNews Title: {}\nNews Articles'
hyde_prompt_template_dict['paq_pairs'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
hyde_prompt_template_dict['s2orc_title_abstract'] = 'Given the following paper title, please write a relevant paper abstract.\nPaper Title: {}\nPaper Abstract:'
hyde_prompt_template_dict['xsum'] = 'Given the following news summary, please write a complete news article.\nNews Summary: {}\nNews Article:'
hyde_prompt_template_dict['zero_shot_re'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:',


hyde_prompt_template_dict['synthetic'] = 'For this retrieval task, write a detailed target passage based on the specified query.\nTask: {instruction}\nQuery: {query}\nPassage:'




cqadup_tasks = []
cqadup_tasks.append('CQADupstackAndroidRetrieval')
cqadup_tasks.append('CQADupstackEnglishRetrieval')
cqadup_tasks.append('CQADupstackGamingRetrieval')
cqadup_tasks.append('CQADupstackGisRetrieval')
cqadup_tasks.append('CQADupstackMathematicaRetrieval')
cqadup_tasks.append('CQADupstackPhysicsRetrieval')
cqadup_tasks.append('CQADupstackProgrammersRetrieval')
cqadup_tasks.append('CQADupstackStatsRetrieval')
cqadup_tasks.append('CQADupstackTexRetrieval')
cqadup_tasks.append('CQADupstackUnixRetrieval')
cqadup_tasks.append('CQADupstackWebmastersRetrieval')
cqadup_tasks.append('CQADupstackWordpressRetrieval')

for cqadup_task in cqadup_tasks:
    hyde_prompt_template_dict[cqadup_task] = \
    'Given the following Stackexchange question, please write a detailed Stackexchange question description.\nQuestion: {}\nDescription:'


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
    parser.add_argument('--max_model_len',type=int,required=True)
    parser.add_argument('--gpu_memory_utilization',type=float,default=0.9)

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
              max_model_len=args.max_model_len, dtype=args.model_dtype, gpu_memory_utilization=args.gpu_memory_utilization,
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

        instructions = None
        if 'synthetic' in task:
            instructions = list(map(lambda x: x['instruction'], query_js_s))

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
            if task not in hyde_prompt_template_dict:
                logger.info('task {} does have a pseudo-psg generation prompt')
                continue
            else:
                hyde_prompt_template = hyde_prompt_template_dict[task]
        print('*'*40)
        print('hyde prompt example:')
        if instructions == None:
            print(hyde_prompt_template.format(queries[0]))
        else:
            print(hyde_prompt_template.format(instruction=instructions[0],query=queries[0]))
        print('*' * 40)
        for qid, q in qid_and_query_s:
            q = ' '.join(q.split(' ')[:args.retrieval_query_max_length])
            if instructions == None:
                hyde_prompt = hyde_prompt_template.format(q)
            else:
                instruction = instructions[qid]
                hyde_prompt = hyde_prompt_template.format(instruction=instruction, query=q)

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
        for i, ((qid, query), llm_prompt) in enumerate(zip(qid_and_query_s, prompts)):
            output_js = {'query': query, 'pseudo_psg': pseudo_passage_s[i], 'qid':qid, 'llm_prompt':llm_prompt}
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



