import argparse
from typing import List, Union, Tuple
import jsonlines
import math
from tqdm import trange
import torch
import pickle
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')


# Save p_embeddings to local position.
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(embeddings, file, protocol=4)


# Load local p_embeddings
def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        embeddings = pickle.load(file)
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


def bge_retrieval(queries: List[str], p_embeddings, query_chunk_size: int=128, top_k: int=100, embedding_chunk_size: int=2) -> List[List]:
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
    global model

    q_embeddings = model.encode_queries(queries, batch_size=query_chunk_size)
    len_p_embeddings = len(p_embeddings)
    chunk_len = math.ceil(len_p_embeddings / embedding_chunk_size)
    docs_list = []
    for start_index in trange(0, len(q_embeddings), query_chunk_size):
        embeddings_batch = q_embeddings[start_index: start_index + query_chunk_size]
        embeddings_batch = torch.tensor(embeddings_batch).cuda()

        all_scores = []
        for i in range(embedding_chunk_size):
            chunk_start_index = i * chunk_len
            chunk_end_index = (i + 1) * chunk_len
            p_embeddings_chunk = torch.tensor(p_embeddings[chunk_start_index: chunk_end_index]).cuda()
            chunk_score = torch.matmul(embeddings_batch, p_embeddings_chunk.t())
            all_scores.append(chunk_score)
            del p_embeddings_chunk
            torch.cuda.empty_cache()

        scores = torch.cat(all_scores, dim=1)
        assert scores.shape[1] == len_p_embeddings, "Not the same length."
        # Save top-k documents.
        values, indices = torch.topk(scores, top_k)
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
    for neg, pos_id in zip(negs, pos_ids):
        new_neg = []
        for doc in neg:
            if doc["id"] not in pos_id:
                new_neg.append(doc)
            else:
                cnt += 1
                logger.info(f"{cnt} docs in pos. Current id: {doc['id']}")
        new_negs.append(new_neg)
    return new_negs


def get_pos(file_path: str) -> List[List[str]]:
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
        for d in fin:
            new_positives.append([pos_psg["title"] + "\n\n" + pos_psg["text"] if pos_psg["title"] != "" else pos_psg["text"] for pos_psg in d["positive_passages"]])
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
                    logger.info(f"{cnt} answers in negs. Current id: {doc['id']}")
                    is_pos = True
                    break
            if not is_pos:
                new_neg.append(doc)
        new_negs.append(new_neg)

    return new_negs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find hard negatives.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True, help="For BM25, it will be something like 'msmarco-v1-passage' sometimes if 'use_prebuilt_index' is set to True.")
    parser.add_argument("--use_prebuilt_index", action="store_true", help="Whether to use prebuilt index in LuceneSearcher.")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--file_path", type=str, required=True, help="Path to a jsonl file that contains id, query, answers and so on. i.e. train_positive_pids.jsonl")
    parser.add_argument("--process_num", type=int, default=30, help="How many processes are used for bm25 retrieval")
    parser.add_argument("--query_chunk_size", type=int, default=256, help="How many samples of the current batch are used for retrieval.")
    parser.add_argument("--embedding_chunk_size", type=int, default=2, help="Chunk the embedding into embedding_chunk_size blocks.")
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    queries = get_queries(args.file_path)

    if args.retriever == "bm25":
        specific_model_name = args.retriever
        from mine.V6.pseudo_psg_scoring.bm25_multi_process import BM25MultiProcess
        model = BM25MultiProcess(args.dataset_name, args.corpus_path, top_k=args.top_k, use_prebuilt_index=args.use_prebuilt_index)
        pool = model.start_multi_process_pool(process_num=args.process_num)
        docs_list = model.retrieve_multi_process(queries, pool)
        model.stop_multi_process_pool(pool)
    else:
        model = get_retriever(args.retriever)

        corpus = []
        logger.info("Load local corpus.")
        with jsonlines.open(args.corpus_path) as fin:
            for line in fin:
                id, text = line["id"], line["contents"]
                corpus.append((id, text))

        specific_model_name = args.retriever.split("/")[-1]
        embedding_path = f"./tmp_indexes/{args.dataset_name}_embedding_{specific_model_name}.pkl"
        if os.path.exists(embedding_path):
            logger.info("Load local embeddings.")
            p_embeddings = load_embeddings(embedding_path)
        else:
            logger.info("Build embeddings.")
            p_embeddings = model.encode([item[1] for item in corpus], batch_size=256)
            save_embeddings(p_embeddings, embedding_path)
        print('p_embeddings type: {}'.format(type(p_embeddings)))
        docs_list = bge_retrieval(queries, p_embeddings, query_chunk_size=args.query_chunk_size, top_k=args.top_k, embedding_chunk_size=args.embedding_chunk_size)
        del corpus

    answers = get_answers(args.file_path)

    if args.dataset_name == "squad":
        queries, new_poss, new_negs, answers = get_query_pos_neg_ans(queries, docs_list, answers)
    else:
        pos_ids = get_pos_ids(args.file_path)
        new_negs = remove_pos(docs_list, pos_ids)
        new_poss = get_pos(args.file_path)
        new_negs = remove_neg_with_answer(new_negs, answers)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.dataset_name}_contrastive_learning_{specific_model_name}.jsonl")
    save_data(save_path, queries, new_poss, new_negs, answers)

    logger.info("All done!")
