task_to_query_type = {
    'eli5': 'user question',
    'hotpot_qa': 'multi-hop question',
    'fever': 'claim',
    'msmarco_passage': 'web search query',
    'msmarco_document': 'web search query',
    'nq': 'question',
    'squad': 'question',
    'triviaqa': 'question',
    'quora_duplicate': 'question',
}

task_to_passage_type = {
    'eli5': 'answers',
    'hotpot_qa': 'supporting documents',
    'fever': 'documents that support or refute the claim',
    'msmarco_passage': 'relevant passages that answer the query',
    'msmarco_document': 'relevant documents that answer the query',
    'nq': 'Wikipedia passages that answer the question',
    'squad': 'Wikipedia passages that answer the question',
    'triviaqa': 'Wikipedia passages that answer the question',
    'quora_duplicate': 'questions that have the same meaning as the question',
}