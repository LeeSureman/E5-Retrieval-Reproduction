def get_hyde_prompt_template():
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

    hyde_prompt_template_dict[
        'eli5'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
    hyde_prompt_template_dict[
        'squad'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
    hyde_prompt_template_dict[
        'triviaqa'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
    hyde_prompt_template_dict[
        'quora_duplicate'] = 'Given the following question, please write 10 diverse paraphrased versions of it.\nQuestion: {}\nParaphrases:'
    hyde_prompt_template_dict['QuoraRetrieval'] = hyde_prompt_template_dict['quora_duplicate']

    hyde_prompt_template_dict[
        'agnews'] = 'Given the following news title, please write a corresponding news article.\nNews Title: {}\nNews Articles'
    hyde_prompt_template_dict[
        'amazon_qa'] = 'Given the following Amazon user question, please write a detailed answer for it.\nQuestion: {}\nAnswer:'
    hyde_prompt_template_dict[
        'amazon_review'] = 'Given the following Amazon review title, please write a corresponding review.\nReview Title: {}\nReview:'
    hyde_prompt_template_dict[
        'ccnews_title_text'] = 'Given the following news title, please write a corresponding news article.\nNews Title: {}\nNews Articles'
    hyde_prompt_template_dict[
        'npr'] = 'Given the following news title, please write a corresponding news article.\nNews Title: {}\nNews Articles'
    hyde_prompt_template_dict[
        'paq_pairs'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:'
    hyde_prompt_template_dict[
        's2orc_title_abstract'] = 'Given the following paper title, please write a relevant paper abstract.\nPaper Title: {}\nPaper Abstract:'
    hyde_prompt_template_dict[
        'xsum'] = 'Given the following news summary, please write a complete news article.\nNews Summary: {}\nNews Article:'
    hyde_prompt_template_dict[
        'zero_shot_re'] = 'Given the following question, please write a passage to answer it.\nQuestion: {}\nPassage:',

    hyde_prompt_template_dict[
        'synthetic'] = 'For this retrieval task, write a detailed target passage based on the specified query.\nTask: {instruction}\nQuery: {query}\nPassage:'

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

    return hyde_prompt_template_dict
