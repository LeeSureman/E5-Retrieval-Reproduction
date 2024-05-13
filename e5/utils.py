import torch
import logging
from torch import Tensor
from transformers import PreTrainedTokenizerFast, BatchEncoding
from typing import Mapping, Dict, List


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def create_batch_dict(
    tokenizer: PreTrainedTokenizerFast, 
    input_texts: List[str], 
    always_add_eos: bool,
    max_length: int = 512
) -> BatchEncoding:
    
    if not always_add_eos:
        return tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )
    else:
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True
        )

        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]

        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )


def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], representation_token, representation_token_num,
                         max_length: int = 512) -> BatchEncoding:
    assert tokenizer.add_bos_token
    representation_special_tokens = representation_token * representation_token_num
    representation_special_ids = tokenizer(representation_special_tokens)['input_ids'][1:]

    representation_special_tokens_num = len(representation_special_ids)

    batch_dict = tokenizer(
        input_texts,
        max_length=max_length - representation_special_tokens_num,
        return_token_type_ids=False,
        return_attention_mask=False,
        padding=False,
        truncation=True
    )

    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + representation_special_ids for input_ids in batch_dict['input_ids']]
    return tokenizer.pad(
        batch_dict,
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors="pt",
        max_length=max_length
    )


def get_task_def_by_task_name_and_type(task_name: str, task_type: str) -> str:
    if task_type in ['Retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update({k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    raise ValueError(f"No instruction config for task {task_name} with type {task_type}")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''
    return 'Instruct: {}\nQuery: '.format(task_description)


task_to_query_type = {
    'ArguAna': 'a claim',
    'ClimateFEVER': 'a claim about climate change',
    'DBPedia': 'a query',
    'FEVER': 'a claim',
    'FiQA2018': 'a financial question',
    'HotpotQA': 'a multi-hop question',
    'MSMARCO': 'a web search query',
    'NFCorpus': 'a question',
    'NQ': 'a question',
    'QuoraRetrieval': 'a question',
    'SCIDOCS': 'a scientific paper title',
    'SciFact': 'a scientific claim',
    'Touche2020': 'a question',
    'TRECCOVID': 'a query on COVID-19',
}

task_to_passage_type = {
    'ArguAna': 'documents that refute the claim',
    'ClimateFEVER': 'documents that support or refute the claim',
    'DBPedia': 'relevant entity descriptions from DBPedia',
    'FEVER': 'documents that support or refute the claim',
    'FiQA2018': 'user replies that best answer the question',
    'HotpotQA': 'supporting documents',
    'MSMARCO': 'relevant passages that answer the query',
    'NFCorpus': 'relevant documents that best answer the question',
    'NQ': 'Wikipedia passages that answer the question',
    'QuoraRetrieval': 'questions that have the same meaning as the question',
    'SCIDOCS': 'paper abstracts that are cited by the given paper',
    'SciFact': 'documents that support or refute the claim',
    'Touche2020': 'detailed and persuasive arguments that answer the question',
    'TRECCOVID': 'documents that answer the query',
}


def get_task_instruction_by_task_name(task_name):
    instruction_template = 'Given {}, retrieve {}.'
    if task_name.lower().startswith('cqadupstack'):
        query_type = 'a question'
        passage_type = 'detailed question descriptions from Stackexchange that are duplicates to the given question'
        return instruction_template.format(query_type, passage_type)
    else:
        return instruction_template.format(task_to_query_type[task_name], task_to_passage_type[task_name])
    