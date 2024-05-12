from typing import Mapping, Dict, List

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

    # instruction_template = 'Given the following {}, give me {}.'
    instruction_template = 'Given {}, retrieve {}.'

    if task_name.lower().startswith('cqadupstack'):
        query_type = 'a question'
        passage_type = 'detailed question descriptions from Stackexchange that are duplicates to the given question'
        return instruction_template.format(query_type, passage_type)
    else:
        return instruction_template.format(task_to_query_type[task_name], task_to_passage_type[task_name])

    # if task_name.lower().startswith('cqadupstack'):
    #     return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'
    #
    # task_name_to_instruct: Dict[str, str] = {
    #     'ArguAna': 'Given a claim, find documents that refute the claim',
    #     'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
    #     'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
    #     'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
    #     'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
    #     'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
    #     'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
    #     'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
    #     'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
    #     'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
    #     'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
    #     'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
    #     'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
    #     'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
    #     # C-MTEB eval instructions
    #     'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
    #     'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
    #     'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
    #     'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
    #     'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
    #     'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
    #     'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
    #     'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
    # }