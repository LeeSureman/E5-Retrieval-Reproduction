import argparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import jsonlines
import random
from typing import Callable, List
import re
from tqdm import trange
import ast
import time
import openai
openai.api_key = "Put your OpenAI API key here"

from openai_manager import get_account_manager

TASK_TYPES = ["short-long", "long-long"]
SHORT_LONG_QUERY_TYPE_SET = ["extremely long-tail", "long-tail", "common"]
SHORT_LONG_QUERY_LENGTH_SET = ["less than 5 words", "5-10 words", "at least 10 words"]
SHORT_LONG_DIFFICULTY_SET = ["high school", "college", "PhD"]
SHORT_LONG_CLARITY_SET = ["clear", "understandable with some effort", "ambiguous"]
SHORT_LONG_NUM_WORDS_SET = [50, 100, 200, 300, 400, 500]

SHORT_LONG_BRAINSTORM_TEMPLATE = """Brainstorm a list of potentially useful text retrieval tasks.

Here are a few examples for your reference:
{prompt_tasks}

Please adhere to the following guidelines:
- Specify the type of the query and documents, and the relationship between the desired documents and the query.
- Each retrieval task should not be too specific and must not specify the specific query content.
- These task descriptions should span various domains, lengths and degrees of specificity.
- The list should include some general retrieval tasks akin in specificity to "{general_task}"

Your output should always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. Be creative!"""

LONG_LONG_BRAINSTORM_TEMPLATE = """Brainstorm a list of text matching tasks where the queries are long documents.

Here are a few examples:
{prompt_tasks}

Please adhere to the following guidelines:
- The list should span various domains and relationships between the queries and documents.
- The list should span various degrees of specificity and include some general retrieval tasks akin in specificity to "{general_task}"

Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"""


def wrap_prompt_tasks(prompt_tasks):
    assert type(prompt_tasks) == type([])
    prompt_tasks = list(map(lambda x: '- ' + (x.strip()), prompt_tasks))
    prompt_tasks_str = '\n'.join(prompt_tasks)
    return prompt_tasks_str

def brainstorm(task_type: str, call_llm: Callable, args, prompt_tasks, general_task) -> str:
    logger.info("`prompt_template` is None, call OpenAI API.")
    if task_type == "long-long":
        messages = [{"role": "user", "content": LONG_LONG_BRAINSTORM_TEMPLATE}]
    elif task_type == "short-long":
        logger.info('the task type is short-long.')
        messages = [{"role": "user", "content": SHORT_LONG_BRAINSTORM_TEMPLATE}]
    else:
        raise ValueError(f"{task_type} not supported.")

    if task_type == 'short-long':
        assert prompt_tasks != None
        assert general_task != None
    elif task_type == 'long-long':
        assert prompt_tasks != None
        # assert general_task = None
    if task_type in ['short-long', 'long-long']:
        assert '{prompt_tasks}' in messages[0]['content'], messages[0]['content']
        assert '{general_task}' in messages[0]['content'], messages[0]['content']
        prompt_tasks_str = wrap_prompt_tasks(prompt_tasks)
        messages[0]['content'] = messages[0]['content'].format(prompt_tasks=prompt_tasks_str, general_task=general_task)

    elif task_type == 'long-long':
        assert '{prompt_tasks}' in messages[0]['content'], messages[0]['content']
        prompt_tasks_str = wrap_prompt_tasks(prompt_tasks)
        messages[0]['content'] = messages[0]['content'].format(prompt_tasks=prompt_tasks_str)

    logger.info('sample prompt tasks, the current whole prompt is as:')
    print('*'*20 + 'whole prompt begin' + '*'*20)
    print(messages[0]['content'])
    print('*'*20 + 'whole prompt end' + '*'*20)

    assert '{' not in messages[0]['content'] and '}' not in messages[0]['content'], messages[0]['content']
    is_successful = False
    while not is_successful:
        try:
            brainstormed_tasks = call_llm(model=args.api_model, messages=messages, temperature=1.0, top_p=1.0, max_tokens=600)['choices'][0]['message']['content']
            is_successful = True
        except Exception as e:
            logger.warning("Accessing to OpenAI API failed. Retry after sleeping 5s.")
            logger.info('exception as follows:')
            logger.info(e)
            time.sleep(5)
        
    return brainstormed_tasks


def get_task_type() -> str:
    return random.choice(TASK_TYPES)


def parse_brainstormed_tasks(brainstormed_tasks: str, api_model: str) -> List[str]:
    is_list = False
    if brainstormed_tasks.startswith('```python'):
        brainstormed_tasks = brainstormed_tasks[len('```python'):]
    while brainstormed_tasks.endswith('`'):
        brainstormed_tasks = brainstormed_tasks[:-1]
    try:
        parsed_tasks = ast.literal_eval(brainstormed_tasks)
        if isinstance(parsed_tasks, list):
            is_list = True
    except:
        print("Cannot be directly parsed into json.")

    if is_list is False and ("gpt-3.5-turbo" in api_model or 'gpt-4' in api_model):
        parsed_tasks = [re.sub(r'^- ', '', item) for item in brainstormed_tasks.split("\n") if item not in ["[", "]"]]
    elif is_list is False:
        raise ValueError(f"{api_model} not supported.")
    
    return parsed_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainstorm_num", type=int, required=True)
    parser.add_argument("--api_model", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument('--task_type', choices=TASK_TYPES, required=True)
    parser.add_argument('--sample_prompt_tasks_from', required=True)
    parser.add_argument('--sample_prompt_tasks_num', type=int, default=3)
    args = parser.parse_args()

    account_manager = get_account_manager('synthesize/openai_account/available.txt', 'synthesize/openai_account/used.txt', multi_thread=True)
    call_llm = openai.ChatCompletion.create

    all_prompt_tasks = list(open('{}/{}.txt'.format(args.sample_prompt_tasks_from, args.task_type)).readlines())
    result_each_call_api = []
    for turn in trange(args.brainstorm_num):
        # 1. Get task type
        task_type = args.task_type
        logger.info(f"Turn: {turn}; task type: {task_type}")
        if task_type == 'short-long':
            general_tasks = [
                'Given a claim, retrieve documents that refute the claim.',
                'Given a claim, retrieve documents that support or refute the claim.',
                'Given a question, retrieve passages that answer the question.',
                'Given a web search query, retrieve relevant passages that answer the query.',
            ]
            tmp_general_task = random.choice(general_tasks)
        elif task_type == 'long-long':
            general_tasks = [
                'Given a passage, retrieve documents that refute it.',
                'Given a sentence, retrieve passages which support or refute the sentence.',
                'Given a paper abstract, retrieve other paper that cite the paper.',
            ]
            tmp_general_task = random.choice(general_tasks)
        else:
            tmp_general_task = None
        # 2. Brainstorm
        tmp_prompt_tasks = random.sample(all_prompt_tasks, args.sample_prompt_tasks_num)
        brainstormed_tasks = brainstorm(task_type, call_llm, args, tmp_prompt_tasks, tmp_general_task)
        # 3. Parse brainstormed string tasks to list
        parsed_tasks = parse_brainstormed_tasks(brainstormed_tasks, args.api_model)
        result_each_call_api.append({'parsed_tasks':parsed_tasks, 'llm_original_response':brainstormed_tasks})

    saved_dir = os.path.join(args.save_dir, 'brainstorm')
    os.makedirs(saved_dir, exist_ok=True)
    save_path = f"{saved_dir}/{args.api_model}_{args.task_type}_syn_data.jsonl"
    with jsonlines.open(save_path,'w') as f_out:
        f_out.write_all(result_each_call_api)

