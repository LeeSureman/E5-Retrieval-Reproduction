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



# task_to_query_type = {
#     'eli5': 'user question',
#     'hotpot_qa': 'multi-hop question',
#     'fever': 'claim',
#     'msmarco_passage': 'web search query',
#     'msmarco_document': 'web search query',
#     'nq': 'question',
#     'squad': 'question',
#     'triviaqa': 'question',
#     'quora_duplicate': 'question',
#     'agnews': 'news title',
#     'amazon_qa': 'Amazon user question',
#     'amazon_review': 'Amazon review title',
#     'ccnews_title_text': 'news title',
#     'gooaq_pairs': 'user question', # query单一，且答案池重复多导致false negative多，所以不用
# #     'medmcqa': 'medical question', # 是短问题匹配短答案，质量不高，所以不用
#     'npr': 'news title',
#     'paq_pairs': 'question',
#     's2orc_title_abstract': 'paper title',
#     'simple_wiki': 'Wikipedia sentence', # 太简单，所以不用
#     'wikihow': 'Wikipedia summary', # positive和query不直接相关，所以不用
#     'xsum': 'news summary',
#     'yahoo_answers_title_answer': 'question', # 感觉和msmarco没明显区别，可以不用
#     'zero_shot_re': 'Wikipedia question',
# }
#
# task_to_passage_type = {
#     'eli5': 'answers',
#     'hotpot_qa': 'supporting documents',
#     'fever': 'documents that support or refute the claim',
#     'msmarco_passage': 'relevant passages that answer the query',
#     'msmarco_document': 'relevant documents that answer the query',
#     'nq': 'Wikipedia passages that answer the question',
#     'squad': 'Wikipedia passages that answer the question',
#     'triviaqa': 'Wikipedia passages that answer the question',
#     'quora_duplicate': 'questions that have the same meaning as the question',
#     'agnews': 'news articles that support the news title',
#     'amazon_qa': 'relevant answers',
#     'amazon_review': 'reviews that support the title',
#     'ccnews_title_text': 'articles that support the title',
#     'gooaq_pairs': 'answers',
#     'medmcqa': 'answers',
#     'npr': 'articles that support the title',
#     'paq_pairs': 'passages that answer the question',
#     's2orc_title_abstract': 'paper abstracts relevant to the paper title',
#     'simple_wiki': 'simplified Wikipedia sentences aligning to the given Wikipedia sentence',
#     'wikihow': 'Wikipedia passages that enrich the Wikipedia summary',
#     'xsum': 'news articles that support the news summary',
#     'yahoo_answers_title_answer': 'answers',
#     'zero_shot_re': 'Wikipedia sentences that answer the Wikipedia question',
# }

# task_to_query_type = {
#     'eli5': 'a user question',
#     'hotpot_qa': 'a multi-hop question',
#     'fever': 'a claim',
#     'msmarco_passage': 'a web search query',
#     'msmarco_document': 'a web search query',
#     'nq': 'a question',
#     'squad': 'a question',
#     'triviaqa': 'a question',
#     'quora_duplicate': 'a question',
#     'agnews': 'a news title',
#     'amazon_qa': 'a user question',
#     'amazon_review': 'a title',
#     'ccnews_title_text': 'a news title',
#     'gooaq_pairs': 'a user question',
#     'medmcqa': 'a medical question',
#     'npr': 'a news title',
#     'paq_pairs': 'a question',
#     's2orc_title_abstract': 'a paper title',
#     'simple_wiki': 'a Wikipedia sentence',
#     'wikihow': 'a Wikipedia summary',
#     'xsum': 'a news summary',
#     'yahoo_answers_title_answer': 'a question',
#     'zero_shot_re': 'a Wikipedia question',
#     'fiqa': 'a financial question',
# }

# task_to_passage_type = {
#     'eli5': 'answers',
#     'hotpot_qa': 'supporting documents',
#     'fever': 'documents that support or refute the claim',
#     'msmarco_passage': 'relevant passages that answer the query',
#     'msmarco_document': 'relevant documents that answer the query',
#     'nq': 'Wikipedia passages that answer the question',
#     'squad': 'Wikipedia passages that answer the question',
#     'triviaqa': 'Wikipedia passages that answer the question',
#     'quora_duplicate': 'questions that have the same meaning as the question',
#     'agnews': 'news articles that support the news title',
#     'amazon_qa': 'reviews that answer the question',
#     'amazon_review': 'reviews relevant to the title',
#     'ccnews_title_text': 'news articles that support the news title',
#     'gooaq_pairs': 'answers',
#     'medmcqa': 'answers',
#     'npr': 'news articles that support the news title',
#     'paq_pairs': 'documents that answer the question',
#     's2orc_title_abstract': 'paper abstract relevant to the paper title',
#     'simple_wiki': 'simplified Wikipedia sentences aligning to the given Wikipedia sentence',
#     'wikihow': 'Wikipedia passages that enrich the Wikipedia summary',
#     'xsum': 'news articles that enrich the news summary',
#     'yahoo_answers_title_answer': 'answers',
#     'zero_shot_re': 'Wikipedia sentences that answer the Wikipedia question',
#     'fiqa': 'user replies that best answer the question',
# }

# task_to_query_type = {
#     'cmedqa2': '一个有关医学的问题',
#     'dureader': '一个用户问题',
#     'mmarco_merged': '一个问题',
#     'multi-cpr-ecom': '一个有关电子商务的查询',
#     'multi-cpr-medical': '一个有关医学的问题',
#     'multi-cpr-video': '一个有关娱乐视频的查询',
#     't2ranking': '一个问题',
#     'covid': '一个问题',
# }

# task_to_passage_type = {
#     'cmedqa2': '相关的文章以回答该问题',
#     'dureader': '相关的文章以回答该问题',
#     'mmarco_merged': '相关的文章以回答该问题',
#     'multi-cpr-ecom': '相关的文章以支持该查询',
#     'multi-cpr-medical': '相关的文章以回答该问题',
#     'multi-cpr-video': '相关的文章以支持该查询',
#     't2ranking': '相关的文章以回答该问题',
#     'covid': '相关的文章以回答该问题',
# }