import argparse
from typing import List, Union, Tuple
import jsonlines
import math
from tqdm import trange
import torch
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
import faiss
import json
sys.path.append('..')
sys.path.append('.')


# Save p_embeddings to local position.
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as file:
        torch.save(embeddings, file, protocol=4)


# Load local p_embeddings
def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        embeddings = torch.load(file)
    return embeddings


def get_retriever(retriever: str) -> object:
    """Get specific retriever accordingt to the model name."""
    if retriever == "BAAI/bge-large-en-v1.5":
        from FlagEmbedding import FlagModel
        model = FlagModel(retriever, 
                          query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")
    
    else:
        raise ValueError(f"{retriever} not supported.")
    
    return model  


def get_queries(file_path: str) -> List[str]:
    """Get queries for retrieval.
    Args
    ----
    file_path: str
        Path to dataset.

    Returns
    -------
    queries: List[str]
        All queries for retrieval.
    """
    queries = []
    with jsonlines.open(file_path) as fin:
        for d in fin:
            queries.append(d["query"])   
    return queries


def retrieval(q_embeddings, p_embeddings, corpus, use_cuda, query_chunk_size: int=128, top_k: int=100, embedding_chunk_size: int=2) -> List[List]:
    """Use retriever of bge to do retrieval on the corpus.
    Args
    ----
    queries: List[str]
        Queries for retrieval.
    query_chunk_size: int=128
        How many samples of the current batch are used for retrieval.
    top_k: int=100
        Returns the top k number of passages based on similarity.
    embedding_chunk_size: int=2
        Chunk the embedding into embedding_chunk_size blocks.

    Returns
    -------
    docs_list: List[List]
        Retrieval result.
    """
    assert len(corpus) == len(p_embeddings)
    global model

    # q_embeddings = model.encode_queries(queries, batch_size=query_chunk_size)
    len_p_embeddings = len(p_embeddings)
    chunk_len = math.ceil(len_p_embeddings / embedding_chunk_size)
    docs_list = []

    faiss_index_type = None

    if faiss_index_type == None:
        faiss_index = None
    elif faiss_index_type == 'IndexFlatIP':
        hidden_size = q_embeddings.size()[1]
        faiss_index = faiss.IndexIVFFlat(hidden_size)
        faiss_index.add(p_embeddings.to(torch.float32))


    for start_index in trange(0, len(q_embeddings), query_chunk_size):
        q_embeddings_batch = q_embeddings[start_index: start_index + query_chunk_size]
        if faiss_index == None:
            if use_cuda:
                q_embeddings_batch = q_embeddings_batch.cuda()

            all_scores = []
            for i in range(embedding_chunk_size):
                chunk_start_index = i * chunk_len
                chunk_end_index = (i + 1) * chunk_len
                p_embeddings_chunk = p_embeddings[chunk_start_index: chunk_end_index]
                if use_cuda:
                    p_embeddings_chunk = p_embeddings_chunk.cuda()
                chunk_score = torch.matmul(q_embeddings_batch, p_embeddings_chunk.t())
                all_scores.append(chunk_score)
                del p_embeddings_chunk
                torch.cuda.empty_cache()

            scores = torch.cat(all_scores, dim=1)
            assert scores.shape[1] == len_p_embeddings, "Not the same length."
            # Save top-k documents.
            values, indices = torch.topk(scores, top_k)
        else:
            raise NotImplementedError
            pass


        for i, doc_idx in enumerate(indices):
            docs = []
            for j, idx in enumerate(doc_idx):
                id, text = corpus[idx.item()]
                docs.append({"id": id, "text": text, "score": values[i][j].item()})
            docs_list.append(docs)

    return docs_list


def get_pos_ids(file_path: str) -> List[List[Union[str, int]]]:
    """Get id of positive sample.
    Args
    ----
    file_path: str
        Path to dataset.

    Returns
    -------
    pos_ids: List[List[Union[str, int]]]
        Ids of all positive samples.
    """
    pos_ids = []
    with jsonlines.open(file_path) as fin:
        for d in fin:
            pos_id = d["positive_pids"]
            for item in d["overlap_pids"]:
                pos_id.append(item[0]) # Second element is score.
            pos_ids.append(pos_id)
    return pos_ids


def remove_pos(negs: List[List], pos_ids: List[List[Union[str, int]]]) -> List[List]:
    """Delete positive samples from negative samples.
    Args
    ----
    negs: List[List]
        The passages retrieved by the retriever.  
    pos_ids: List[List[Union[str, int]]]
        Ids of positive samples.

    Returns
    -------
    new_negs: List[List]
        Negative samples without positive samples.
    """
    new_negs = []
    cnt = 0
    all_pos_num = 0
    for neg, pos_id in zip(negs, pos_ids):
        all_pos_num+=len(pos_id)
        new_neg = []
        for doc in neg:
            if doc["id"] not in pos_id:
                new_neg.append(doc)
            else:
                cnt += 1
                # logger.info(f"{cnt} docs in pos. Current id: {doc['id']}")
        new_negs.append(new_neg)
    print('total pos num: {}'.format(all_pos_num))
    logger.info(f"{cnt} docs in pos. Current id: {doc['id']}")
    return new_negs


def get_pos_id(file_path: str) -> List[List[str]]:
    """Get all positive passages.
    Args
    ----
    file_path: str
        Path to dataset.

    Returns
    -------
    new_positives: List[List[str]]
        Positive passages of all samples.
    """
    new_positives = []
    with jsonlines.open(file_path) as fin:
        for tmp_js in fin:
            new_positives.append(tmp_js['positive_pids'])
            # new_positives.append([pos_psg["title"] + "\n\n" + pos_psg["text"] if pos_psg["title"] != "" else pos_psg["text"] for pos_psg in d["positive_passages"]])
    return new_positives


def save_data(save_path: str, queries: List[str], poss: List[List[str]], negs: List[List], answers: List[List[str]]) -> None:
    """Save data for training.
    Args
    ----
    save_path: str
        Path to save the data.
    queries: List[str]
        Query of all samples.
    poss: List[List[str]]
        Positive passages of all samples.
    negs: List[List]
        Negative passages of all samples.
    answers: List[List[str]]
        The answer to the query.
    """
    with jsonlines.open(save_path, "w") as f:
        for query, pos, neg, ans in zip(queries, poss, negs, answers):
            f.write({
                "query": query,
                "positive": pos,
                "negative": [doc["text"] for doc in neg],
                "answers": ans
            })


def get_answers(file_path: str) -> List[List[str]]:
    """Get answers (ground truth）to help find positive passages.
    Args
    ----
    file_path: str
        Path to dataset.

    Returns
    -------
    new_positives: List[List[str]]
        Positive passages of all samples.
    """
    with jsonlines.open(file_path) as fin:
        answers = [d["answers"] for d in fin]
    return answers


def get_query_pos_neg_ans(
        queries: List[str], 
        docs_list: List[List], 
        answers: List[List[str]]
    ) -> Tuple[List[str], List[List[str]], List[List], List[List[str]]]:
    """Get new queries, positive passages and negative passages.
    Args
    ----
    queries: List[str]
        All queries for retrieval
    docs_list: List[List]
        Retrieval result.
    answers: List[List[str]]
        Answers to the question.

    Returns
    -------
    new_queries: List[str]
        New queries after screening
    new_poss: List[List[str]]
        New positive passages after screening.
    new_negs: List[List]
        New negative passages after screening.
    new_anss: List[List[str]]
        New answers after screening.
    """
    new_queries, new_poss, new_negs, new_anss = [], [], [], []
    cnt = 0
    for query, docs, answer in zip(queries, docs_list, answers):
        poss = []
        negs = []
        for doc in docs:
            is_pos = False
            for answer_text in answer:
                if answer_text.lower() in doc["text"].lower():
                    cnt += 1
                    logger.info(f"{cnt} docs in pos. Current id: {doc['id']}")
                    poss.append(doc["text"])
                    is_pos = True
                    break
            if not is_pos:
                negs.append(doc)
        
        if len(poss) > 0:
            new_queries.append(query)
            new_poss.append(poss)
            new_negs.append(negs)
            new_anss.append(answer)

    return new_queries, new_poss, new_negs, new_anss


def remove_neg_with_answer(negs: List[List], answers: List[List[str]]) -> List[List]:
    """Remove passages with answer.
    Args
    ----
    negs: List[List]
        Negative passages after screening.
    answers: List[List[str]]
        Answers to the question.

    Returns
    -------
    new_negs: List[List]
        New negative samples after removing answers.
    """
    new_negs = []
    cnt = 0
    for neg, answer in zip(negs, answers):
        new_neg = []
        for doc in neg:
            is_pos = False
            for answer_text in answer:
                if answer_text.lower() in doc["text"].lower():
                    cnt += 1
                    # logger.info(f"{cnt} answers in negs. Current id: {doc['id']}")
                    is_pos = True
                    break
            if not is_pos:
                new_neg.append(doc)
        new_negs.append(new_neg)
    logger.info(f"{cnt} answers in negs. Current id: {doc['id']}")

    return new_negs

def remove_neg_same_as_query(negs_list, queries):
    new_negs_list = []
    cnt=0
    for negs, q in zip(negs_list, queries):
        new_negs = []
        q = q.lower()
        for neg in negs:
            if q in neg['text'].lower():
                cnt += 1
                # logger.info(f"{cnt} queries in negs. Current id: {neg['id']}")
                continue
            else:
                new_negs.append(neg)
        new_negs_list.append(new_negs)
    logger.info(f"{cnt} queries in negs. Current id: {neg['id']}")

    return new_negs_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find hard negatives.")
    # parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    # parser.add_argument("--file_path", type=str, required=True, help="Path to a jsonl file that contains id, query, answers and so on. i.e. train_positive_pids.jsonl")
    parser.add_argument('--metadata_fp',required=True)
    parser.add_argument("--process_num", type=int, default=30, help="How many processes are used for bm25 retrieval")
    parser.add_argument("--query_chunk_size", type=int, default=256, help="How many samples of the current batch are used for retrieval.")
    # parser.add_argument("--embedding_chunk_size", type=int, default=2, help="Chunk the embedding into embedding_chunk_size blocks.")
    parser.add_argument('--max_p_embedding_batch',type=int,required=True)
    parser.add_argument('--p_embed_dir',required=True)
    parser.add_argument('--q_embed_dir',required=True)
    parser.add_argument('--output_fp',required=True)
    parser.add_argument('--corpus_path',required=True)
    parser.add_argument('--use_cuda',required=True)
    parser.add_argument('--extra_q_to_pos_fp',)

    args = parser.parse_args()

    metadata_js_s = list(jsonlines.open(args.metadata_fp))
    q_to_metadata = {}
    for tmp_js in metadata_js_s:
        q_to_metadata[tmp_js['query']] = tmp_js

    corpus = []
    logger.info("Load local corpus.")
    with jsonlines.open(args.corpus_path) as fin:
        for line in fin:
            id, text = line["id"], line["contents"]
            corpus.append((id, text))

    print('corpus: {}'.format(len(corpus)))

    if False:
        pass
    else:


        p_embed_filename_s = os.listdir(args.p_embed_dir)
        p_embed_js_s = []
        for fname in p_embed_filename_s:
            fp = os.path.join(args.p_embed_dir, fname)
            js_s_shard = torch.load(open(fp, 'rb'))
            p_embed_js_s.extend(js_s_shard)
        p_embed_js_s.sort(key=lambda x:x['gidx'])

        p_embeddings = list(map(lambda x:x['embed'].unsqueeze(0), p_embed_js_s))
        p_embeddings = torch.cat(p_embeddings, dim=0)
        print('p_embeddings: {}'.format(p_embeddings.size()))
        assert len(corpus) == p_embeddings.size()[0], "{} {}".format(len(corpus), p_embeddings.size())

        args.embedding_chunk_size = (p_embeddings.size()[0] // args.max_p_embedding_batch) +1

        q_embeds_filename_s = os.listdir(args.q_embed_dir)
        q_embeds_js_s = []
        for fname in q_embeds_filename_s:
            fp = os.path.join(args.q_embed_dir, fname)
            js_s_shard = torch.load(open(fp, 'rb'))
            q_embeds_js_s.extend(js_s_shard)
        q_embeds_js_s.sort(key=lambda x: x['gidx'])

        # q_embeds_js_s = q_embeds_js_s[:200]

        queries = list(map(lambda x:x['query'], q_embeds_js_s))

        num_embeds_each_q = len(q_embeds_js_s[0]['embeds'])

        q_embeddings = []
        for tmp_js in q_embeds_js_s:
            q_embeddings.extend(list(map(lambda x:x.unsqueeze(0),tmp_js['embeds'])))

        q_embeddings = torch.cat(q_embeddings, dim=0)
        print('q_embeddings: {}'.format(q_embeddings.size()))



        docs_list_before_filter_cache_fp = '{}.before_filter'.format(args.output_fp)
        if os.path.exists(docs_list_before_filter_cache_fp):
            docs_list = list(jsonlines.open(docs_list_before_filter_cache_fp))

        else:
            docs_list = retrieval(q_embeddings, p_embeddings, corpus=corpus, use_cuda=args.use_cuda, query_chunk_size=args.query_chunk_size, top_k=args.top_k, embedding_chunk_size=args.embedding_chunk_size)
            with jsonlines.open(docs_list_before_filter_cache_fp, 'w') as f_out:
                f_out.write_all(docs_list)
        del corpus

        merged_docs_list = []

        for i, tmp_js in enumerate(q_embeds_js_s):
            start = i * num_embeds_each_q
            end = (i+1) * num_embeds_each_q
            tmp_docs_list = docs_list[start:end]
            all_doc_js_s = []
            all_doc_ids = set()
            for tmp_docs in tmp_docs_list:
                for tmp_doc in tmp_docs:
                    if tmp_doc['id'] in all_doc_ids:
                        continue
                    else:
                        del tmp_doc['score']
                        all_doc_js_s.append(tmp_doc)
                        all_doc_ids.add(tmp_doc['id'])

            assert len(all_doc_js_s) == len(all_doc_ids)
            merged_docs_list.append(all_doc_js_s)

            # tmp_doc_ids = list(map(lambda x:x['id'], tmp_docs_list))
            # tmp_doc_ids = list(set(tmp_doc_ids))





    answers_s = []
    for q in queries:
        answers = q_to_metadata[q]['answers']
        answers_s.append(answers)

    if False:
        pass
        # queries, new_poss, new_negs, answers = get_query_pos_neg_ans(queries, merged_docs_list, answers)
    else:
        pos_ids_s = []
        pos_and_overlap_ids_s = []
        for q in queries:
            tmp_js = q_to_metadata[q]
            pos_ids = tmp_js['positive_pids']
            pos_ids_s.append(pos_ids)
            overlap_pids = list(map(lambda x:x[0], tmp_js["overlap_pids"]))

            pos_and_overlap_ids_s.append(pos_ids + overlap_pids)

        print('pos_and_overlap_ids_s: {}'.format(pos_and_overlap_ids_s[0]))

        new_negs_s = remove_pos(merged_docs_list, pos_and_overlap_ids_s)
        new_negs_s = remove_neg_with_answer(new_negs_s, answers_s)
        new_negs_s = remove_neg_same_as_query(new_negs_s, queries)

        if args.extra_q_to_pos_fp != None:
            new_new_negs_s = []
            q_to_poss_dict = json.load(open(args.extra_q_to_pos_fp))
            new_poss_list = []
            for q in queries:
                new_poss_list.append(q_to_poss_dict[q])

            for negs, new_poss in zip(new_negs_s, new_poss_list):
                negs_num_before = len(negs)
                negs = list(filter(lambda x:x['text'] not in new_poss, negs))
                negs_num_after = len(negs)

                if negs_num_after < negs_num_before:
                    print('filter {} negs according to {}'.format(negs_num_before - negs_num_after, args.extra_q_to_pos_fp))
                new_new_negs_s.append(negs)

            new_negs_s = new_new_negs_s




        # new_negs_s =


    output_js_s = []
    assert len(queries) == len(pos_ids_s)
    assert len(queries) == len(new_negs_s)

    for q, pos_ids, neg_docs in zip(queries,pos_ids_s,new_negs_s):
        tmp_js_output = {}
        tmp_js_output['query'] = q
        tmp_js_output['positive_pids'] = pos_ids
        tmp_js_output['negative_pids'] = list(map(lambda x:x['id'], neg_docs))
        output_js_s.append(tmp_js_output)

    with jsonlines.open(args.output_fp, 'w') as f_out:
        f_out.write_all(output_js_s)


    # save_data(save_path, queries, new_poss, new_negs_s, answers)

    logger.info("All done!")
