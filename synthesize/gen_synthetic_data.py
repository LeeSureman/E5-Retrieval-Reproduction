import argparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import jsonlines
import random
from typing import Callable, List, Dict
import re
import json
import concurrent.futures
import uuid
import tqdm
import ast
import uuid
import openai
openai.api_key = "Put your OpenAI API key here"

from openai_manager import get_account_manager, call_chatgpt

TASK_TYPES = ["short-long", "long-long"]
SHORT_LONG_QUERY_TYPE_SET = ["extremely long-tail", "long-tail", "common"]
SHORT_LONG_QUERY_LENGTH_SET = ["less than 5 words", "5-10 words", "at least 10 words"]
SHORT_LONG_DIFFICULTY_SET = ["high school", "college", "PhD"]
SHORT_LONG_CLARITY_SET = ["clear", "understandable with some effort", "ambiguous"]
SHORT_LONG_NUM_WORDS_SET = [50, 100, 200, 300, 400, 500]

SHORT_LONG_GENERATE_TEMPLATE = """Here is a retrieval task: {task}

Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:
- "user_query": a string, a random user search query specified by the retrieval task.
- "positive_document": a string, a relevant document for the user query.
- "hard_negative_document": a string, a hard negative document that only appears relevant to the query but is actually useless and unhelpful to the query.

Please adhere to the following guidelines:
- The "user_query" should be {query_type}, {query_length}, {clarity}, and diverse in topic.
- All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable if some parts of the "positive_document" are not topically related to the query.
- All documents should be at least {num_words} words long.
- The "hard_negative_document" should be literally relevant with the query but is not the correct target of the query.
- Do not provide any explanation in any document on why it is relevant or not relevant to the query.
- Both the query and documents require {difficulty} level education to understand.

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""

LONG_LONG_GENERATE_TEMPLATE = """You have been assigned a text matching task: {task}

Your mission is to write one example for this task in JSON format. The JSON object must contain the following keys:
- "input": a string, a random input specified by the task, which is long (at least 500 words) and detailed.
- "positive_document": a string, a relevant document for the "input" according to the task.
- "hard_negative_document": a string, a hard negative document that only appears relevant to the query but is actually useless and unhelpful to the query.

Please adhere to the following guidelines:
- The "input", "positive_document" and "hard_negative_document" should be long documents (each of them must at least 500 words), avoid substantial word overlaps, otherwise the task would be too easy.
- All documents must be created independent of the "input" as they and the "input" are not in the same context. Avoid copying the "input" verbatim. It’s acceptable if some parts of the "positive_document" are not topically related to the "input".
- The "hard_negative_document" should be literally relevant with the query but is not the correct target of the query according to this task.
- Do not provide any explanation in any document on why it is relevant or not relevant to the "input".

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""


def generate_examples(task_type: str, call_llm: Callable, parsed_tasks: List[str]) -> List[Dict]:
    examples = []
    if task_type == "long-long":
        for parsed_task in parsed_tasks:
            prompt = LONG_LONG_GENERATE_TEMPLATE.format(task=parsed_task)
            examples.append({"prompt": prompt, "task_type": task_type, "task": parsed_task})

    elif task_type == "short-long":
        for parsed_task in parsed_tasks:
            query_type = random.choice(SHORT_LONG_QUERY_TYPE_SET)
            query_length = random.choice(SHORT_LONG_QUERY_LENGTH_SET)
            clarity = random.choice(SHORT_LONG_CLARITY_SET)
            num_words = random.choice(SHORT_LONG_NUM_WORDS_SET)
            difficulty = random.choice(SHORT_LONG_DIFFICULTY_SET)
            prompt = SHORT_LONG_GENERATE_TEMPLATE.format(task=parsed_task,
                                                        query_type=query_type,
                                                        query_length=query_length,
                                                        clarity=clarity,
                                                        num_words=num_words,
                                                        difficulty=difficulty)
            examples.append({"prompt": prompt, "task_type": task_type, "task": parsed_task,})

    else:
        raise ValueError(f"{task_type} not supported.")
    
    global account_manager
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for example in examples:
            future = executor.submit(
                call_chatgpt,
                call_llm,
                args.api_model,
                example,
                thread_id=uuid.uuid4(),
                account_manager=account_manager,
            )
            futures.append(future)
            print('*' * 20 + 'whole prompt begin' + '*' * 20)
            print(example['prompt'])
            print('*' * 20 + 'whole prompt end' + '*' * 20)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(examples)):
            future.result()

    return examples


def get_task_type() -> str:
    return random.choice(TASK_TYPES)

def parse_brainstormed_tasks(brainstormed_tasks: str, api_model: str) -> List[str]:
    is_list = False
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


def parse_new_examples(new_examples: List[Dict], task_type: str) -> List[Dict]:
    for new_example in new_examples:
        new_example['llm_response'] = new_example["example"]
        try:
            new_example["example"] = json.loads(new_example["example"])
        except:
            new_example["is_valid"] = False
            logger.warning("Cannot parse string to json.")
            continue

        if task_type == "short-long":
            if "user_query" in new_example["example"] and "positive_document" in new_example["example"] and "hard_negative_document" in new_example["example"]:
                new_example["is_valid"] = True
            else:
                new_example["is_valid"] = False
                logger.warning("Json keys error.")
        elif task_type == "long-long":
            if "input" in new_example["example"] and "positive_document" in new_example["example"] and "hard_negative_document" in new_example["example"]:
                new_example["is_valid"] = True
                new_example["example"]['user_query'] = new_example["example"]['input']
                del new_example["example"]['input']
            else:
                new_example["is_valid"] = False
                logger.warning("Json keys error.")
        else:
            raise ValueError(f"{task_type} not supported.")
    
    parsed_examples = [example for example in new_examples if example["is_valid"]]
    return parsed_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", type=str, required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--num_examples_each_task', type=int, required=True)
    parser.add_argument('--task_type', required=True)
    args = parser.parse_args()

    account_manager = get_account_manager('synthesize/openai_account/available.txt', 'synthesize/openai_account/used.txt', multi_thread=True)
    call_llm = openai.ChatCompletion.create

    tasks = list(jsonlines.open(args.input_path))
    result_examples = []
    llm_responses = []
    for i in range(args.num_examples_each_task):
        new_examples = generate_examples(args.task_type, call_llm, tasks)
        llm_responses.extend(new_examples)
        parsed_examples = parse_new_examples(new_examples, args.task_type)
        result_examples.extend(parsed_examples)

    output_dir = os.path.join('synthesize/synthetic_data/generate', args.api_model + '_' + args.task_type)
    os.makedirs(output_dir, exist_ok=True)
    example_output_fp = '{}/examples.jsonl'.format(output_dir)
    with jsonlines.open(example_output_fp, 'w') as f_out:
        f_out.write_all(result_examples)
        logger.info('final example output to {}'.format(example_output_fp))

    llm_response_output_fp = '{}/llm_response.jsonl'.format(output_dir)
    with jsonlines.open(llm_response_output_fp, 'w') as f_out:
        f_out.write_all(llm_responses)
        logger.info('llm response output to {}'.format(llm_response_output_fp))
