task_to_sample_weight = {
    'synthetic': 1,
    'eli5': 1,
    'hotpot_qa': 1,
    'fever': 1,
    'msmarco_passage': 1,
    'nq': 1,
    'squad': 1,
    'triviaqa': 1,
    'fiqa': 1,
    'quora_duplicate': 1,
    'demo': 1,
}

task_to_query_type = {
    'synthetic': 'a user question',
    'eli5': 'a user question',
    'hotpot_qa': 'a multi-hop question',
    'fever': 'a claim',
    'msmarco_passage': 'a web search query',
    'nq': 'a question',
    'squad': 'a question',
    'triviaqa': 'a question',
    'quora_duplicate': 'a question',
    'fiqa': 'a financial question',
    'demo': 'a user question',
}

task_to_passage_type = {
    'synthetic': 'answers',
    'eli5': 'answers',
    'hotpot_qa': 'supporting documents',
    'fever': 'documents that support or refute the claim',
    'msmarco_passage': 'relevant passages that answer the query',
    'nq': 'Wikipedia passages that answer the question',
    'squad': 'Wikipedia passages that answer the question',
    'triviaqa': 'Wikipedia passages that answer the question',
    'quora_duplicate': 'questions that have the same meaning as the question',
    'fiqa': 'user replies that best answer the question',
    'demo': 'answers',
}
