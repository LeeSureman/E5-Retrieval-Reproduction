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
from tqdm import trange
import concurrent.futures
import uuid
import ast
import time
import uuid
import openai
openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_key = "sk-jeBw0RaynjHgfl7fCYQHzuQVCfIoCnJJ7UQGdJUTk6AeKBan"

# from call_api import call_xwin
from openai_manager import get_account_manager, call_chatgpt

(
    xwin_prompt_template := "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            "USER: {user_msg} "
            "ASSISTANT:"
)

# TASK_TYPES = ["short-long", "short-short", "long-long"]
TASK_TYPES = ["short-long", "long-long"]

SHORT_LONG_QUERY_TYPE_SET = ["extremely long-tail", "long-tail", "common"]
SHORT_LONG_QUERY_LENGTH_SET = ["less than 5 words", "5-10 words", "at least 10 words"]
SHORT_LONG_DIFFICULTY_SET = ["high school", "college", "PhD"]
SHORT_LONG_CLARITY_SET = ["clear", "understandable with some effort", "ambiguous"]
SHORT_LONG_NUM_WORDS_SET = [50, 100, 200, 300, 400, 500]

SHORT_LONG_BRAINSTORM_TEMPLATE = """Brainstorm a list of potentially useful text retrieval tasks.

Here are a few examples for your reference:
 - Provided a scientific claim as query, retrieve documents that help verify or refute the claim.
 - Search for documents that answers a FAQ-style query on children's nutrition.

Please adhere to the following guidelines:
 - Specify what the query is, and what the desired documents are.
 - Each retrieval task should cover a wide range of queries, and should not be too specific.

Your output should always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. Be creative!"""

SHORT_LONG_GENERATE_TEMPLATE = """You have been assigned a retrieval task: {task}

Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:
- "user_query": a string, a random user search query specified by the retrieval task.
- "positive_document": a string, a relevant document for the user query.
- "hard_negative_document": a string, a hard negative document that only appears relevant to the query.

Please adhere to the following guidelines:
- The "user_query" should be {query_type}, {query_length}, {clarity}, and diverse in topic.
- All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable if some parts of the "positive_document" are not topically related to the query.
- All documents should be at least {num_words} words long.
- The "hard_negative_document" contains some useful information, but it should be less useful or comprehensive compared to the "positive_document".
- Both the query and documents should be in English.
- Do not provide any explanation in any document on why it is relevant or not relevant to the query.
- Both the query and documents require {difficulty} level education to understand.

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""


SHORT_SHORT_BRAINSTORM_TEMPLATE = """Brainstorm a list of text matching tasks where both the queries and the groundtruth documents are very short (one or two sentences, even a short phrase).

Here are a few examples:
- Given a scientific paper title, retrieve the title of papers that cite the given paper.
- Match a word with its definition.
- Provided a notable person’s name, identify their occupation or achievement.

Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"""

SHORT_SHORT_GENERATE_TEMPLATE = """You have been assigned a text matching task: {task}

Your mission is to write one example for this task in JSON format. The JSON object must contain the following keys:
- "input": a string, a random input specified by the task.
- "positive_document": a string, a relevant document for the "input" according to the task.

Please adhere to the following guidelines:
- The values of all fields should be in English.
- Both the "input" and "positive_document" should be very short (a sentence or a phrase), avoid substantial word overlaps, otherwise the task would be too easy.
- The "input" and "positive_document" should be independent of each other.

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""


LONG_LONG_BRAINSTORM_TEMPLATE = """Brainstorm a list of text matching tasks where the queries are long documents.

Here are a few examples:
- Given a document that supports a debatable argument, find another document that contains opposite arguments.
- Provided a lengthy business proposal, retrieve competitive business strategies in the same industry.

Your output must always be a python list of strings only, with about 20 elements, and each element corresponds to a distinct task in one sentence. Do not explain yourself or output anything else. Be creative!"""

LONG_LONG_GENERATE_TEMPLATE = """You have been assigned a text matching task: {task}

Your mission is to write one example for this task in JSON format. The JSON object must contain the following keys:
- "input": a string, a random input specified by the task.
- "positive_document": a string, a relevant document for the "input" according to the task.
- "hard_negative_document": a string, a hard negative document that only appears relevant to the "input".

Please adhere to the following guidelines:
- The values of all fields should be in English.
- The "input", "positive_document" and "hard_negative_document" should be long documents (at least 300 words), avoid substantial word overlaps, otherwise the task would be too easy.
- All documents must be created independent of the "input". Avoid copying the "input" verbatim. It’s acceptable if some parts of the "positive_document" are not topically related to the "input".

Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!"""


def brainstorm(task_type: str, call_llm: Callable, prompt_template: str) -> str:
    if prompt_template is not None:
        if task_type == "long-long":
            prompt = prompt_template.format(user_msg=LONG_LONG_BRAINSTORM_TEMPLATE)
        elif task_type == "short-long":
            prompt = prompt_template.format(user_msg=SHORT_LONG_BRAINSTORM_TEMPLATE)
        elif task_type == "short-short":
            prompt = prompt_template.format(user_msg=SHORT_SHORT_BRAINSTORM_TEMPLATE)
        else:
            raise ValueError(f"{task_type} not supported.")
        brainstormed_tasks = call_llm(prompt, max_new_tokens=3000)[1]
    else:
        logger.info("`prompt_template` is None, call OpenAI API.")
        if task_type == "long-long":
            messages = [{"role": "user", "content": LONG_LONG_BRAINSTORM_TEMPLATE}]
        elif task_type == "short-long":
            messages = [{"role": "user", "content": SHORT_LONG_BRAINSTORM_TEMPLATE}]
        elif task_type == "short-short":
            messages = [{"role": "user", "content": SHORT_SHORT_BRAINSTORM_TEMPLATE}]
        else:
            raise ValueError(f"{task_type} not supported.")
        
        is_successful = False
        while not is_successful:
            try:
                brainstormed_tasks = call_llm(model="gpt-3.5-turbo-0125", messages=messages, temperature=1.0, top_p=1.0, max_tokens=4096)['choices'][0]['message']['content']
                is_successful = True
            except:
                logger.warning("Accessing to OpenAI API failed. Retry after sleeping 5s.")
                time.sleep(5)
        
    return brainstormed_tasks


def generate(task_type: str, call_llm: Callable, prompt_template: str, parsed_tasks: List[str]) -> List[Dict]:
    examples = []
    if task_type == "long-long":
        for parsed_task in parsed_tasks:
            long_long_generate_tempalte = LONG_LONG_BRAINSTORM_TEMPLATE.format(task=parsed_task)
            if prompt_template is not None:
                prompt = prompt_template.format(user_msg=long_long_generate_tempalte)
            else:
                prompt = long_long_generate_tempalte
            examples.append({"prompt": prompt, "task_type": task_type, "task": parsed_task})

    elif task_type == "short-long":
        for parsed_task in parsed_tasks:
            query_type = random.choice(SHORT_LONG_QUERY_TYPE_SET)
            query_length = random.choice(SHORT_LONG_QUERY_LENGTH_SET)
            clarity = random.choice(SHORT_LONG_CLARITY_SET)
            num_words = random.choice(SHORT_LONG_NUM_WORDS_SET)
            difficulty = random.choice(SHORT_LONG_DIFFICULTY_SET)
            short_long_generate_template = SHORT_LONG_GENERATE_TEMPLATE.format(task=parsed_task,
                                                                               query_type=query_type,
                                                                               query_length=query_length,
                                                                               clarity=clarity,
                                                                               num_words=num_words,
                                                                               difficulty=difficulty)
            if prompt_template is not None:
                prompt = prompt_template.format(user_msg=short_long_generate_template)
            else:
                prompt = short_long_generate_template
            examples.append({"prompt": prompt, "task_type": task_type, "task": parsed_task,})

    elif task_type == "short-short":
        for parsed_task in parsed_tasks:
            short_short_generate_template = SHORT_SHORT_GENERATE_TEMPLATE.format(task=parsed_task)
            if prompt_template is not None:
                prompt = prompt_template.format(user_msg=short_short_generate_template)
            else:
                prompt = short_short_generate_template
            examples.append({"prompt": prompt, "task_type": task_type, "task": parsed_task,})

    else:
        raise ValueError(f"{task_type} not supported.")
    
    if prompt_template is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for index, example in enumerate(examples):
                future = executor.submit(
                    call_llm,
                    prompt=example["prompt"],
                    max_new_tokens=3000,
                    index = index
                )
                futures.append(future)

            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        results = sorted(results, key=lambda x: x[0])
        for example, result in zip(examples, results):
            example["example"] = result[1] # Raw text before parsing
    
    else:
        global account_manager
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for example in examples:
                future = executor.submit(
                    call_chatgpt,
                    call_llm,
                    "gpt-3.5-turbo-0125",
                    example,
                    thread_id=uuid.uuid4(),
                    account_manager=account_manager,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result()

    for example in examples:
        example.pop("prompt")
    return examples


def get_task_type() -> str:
    return random.choice(TASK_TYPES)


def get_model_api(api_model: str) -> Callable:
    if api_model == "xwin-70b-v0.1":
        return call_xwin
    else:
        raise ValueError(f"{api_model} not supported.")
    

def get_prompt_template(api_model: str) -> str:
    if "gpt-3.5-turbo" in api_model:
        return None
    elif api_model == "xwin-70b-v0.1":
        return xwin_prompt_template
    else:
        raise ValueError(f"{api_model} not supported.")


def parse_brainstormed_tasks(brainstormed_tasks: str, api_model: str) -> List[str]:
    is_list = False
    try:
        parsed_tasks = ast.literal_eval(brainstormed_tasks)
        if isinstance(parsed_tasks, list):
            is_list = True
    except:
        print("Cannot be directly parsed into json.")

    if is_list is False and api_model == "xwin-70b-v0.1":
        parsed_tasks = [re.sub(r'^\d+\.\s*', '', item).replace('"', "") for item in brainstormed_tasks.split("\n") if item not in ["[", "]"]]
    elif is_list is False and "gpt-3.5-turbo" in api_model:
        parsed_tasks = [re.sub(r'^- ', '', item) for item in brainstormed_tasks.split("\n") if item not in ["[", "]"]]
    elif is_list is False:
        raise ValueError(f"{api_model} not supported.")
    
    return parsed_tasks


def parse_new_examples(new_examples: List[Dict], task_type: str) -> List[Dict]:
    for new_example in new_examples:
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
        elif task_type == "short-short":
            if "input" in new_example["example"] and "positive_document" in new_example["example"]:
                new_example["is_valid"] = True
            else:
                new_example["is_valid"] = False
                logger.warning("Json keys error.")
        elif task_type == "long-long":
            if "input" in new_example["example"] and "positive_document" in new_example["example"] and "hard_negative_document" in new_example["example"]:
                new_example["is_valid"] = True
            else:
                new_example["is_valid"] = False
                logger.warning("Json keys error.")
        else:
            raise ValueError(f"{task_type} not supported.")
    
    parsed_examples = [example for example in new_examples if example["is_valid"]]
    return parsed_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainstorm_num", type=int, required=True)
    parser.add_argument("--api_model", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    save_path = f"{args.save_dir}/{args.api_model}_syn_data.jsonl"
    if os.path.exists(save_path) and not args.overwrite:
        logger.info("Result file already exists and `overwrite` argument is set to False. The results will be added to the tail.")

    if os.path.exists(save_path) and args.overwrite:
        os.remove(save_path)
        logger.info("Previous result file has been removed.")

    if "gpt-3.5-turbo" in args.api_model:
        account_manager = get_account_manager('openai_account/available.txt', 'openai_account/used.txt', multi_thread=True)
        call_llm = openai.ChatCompletion.create
    else:
        call_llm = get_model_api(args.api_model)
    prompt_template = get_prompt_template(args.api_model)

    for turn in trange(args.brainstorm_num):
        # 1. Get task type
        task_type = get_task_type()
        logger.info(f"Turn: {turn}; task type: {task_type}")
        # 2. Brainstorm
        brainstormed_tasks = brainstorm(task_type, call_llm, prompt_template)
        # 3. Parse brainstormed string tasks to list
        parsed_tasks = parse_brainstormed_tasks(brainstormed_tasks, args.api_model)
        # 4. Generate
        new_examples = generate(task_type, call_llm, prompt_template, parsed_tasks)
        # 5. Parse new string example to json
        parsed_examples = parse_new_examples(new_examples, task_type)
        # 6. Save valid examples
        with jsonlines.open(save_path, "a") as fout:
            fout.write_all(parsed_examples)
