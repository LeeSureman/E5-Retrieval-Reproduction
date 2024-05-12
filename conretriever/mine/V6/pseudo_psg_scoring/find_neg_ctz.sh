# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name msmarco_psg

# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name msmarco_doc

# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name quora

# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name hotpotqa

# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name fever

# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name nq

# python dense_retriever_find_neg.py \
#     --model_name BAAI/bge-large-en-v1.5 \
#     --task_name triviaqa
    
# python bm25_find_neg.py \
#     --task_name quora

# python bm25_find_neg.py \
#     --task_name msmarco_doc

# python bm25_find_neg.py \
#     --task_name msmarco_psg

# python bm25_find_neg.py \
#     --task_name hotpotqa

# python bm25_find_neg.py \
#     --task_name fever

# python bm25_find_neg.py \
#     --task_name nq

# python bm25_find_neg.py \
#     --task_name triviaqa

# python bm25_find_neg.py \
#     --task_name squad

###############################################################
# For pipeline
# python find_neg_pipeline.py \
#     --dataset_name msmarco_doc \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/msmarco_doc/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/msmarco_doc/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name squad \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/squad/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/squad/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 4 \
#     --query_chunk_size 128

# python find_neg_pipeline.py \
#     --dataset_name tevatron_msmarco_psg \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/tevatron_msmarco_psg/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/tevatron_msmarco_psg/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 2 \
#     --query_chunk_size 256

# python find_neg_pipeline.py \
#     --dataset_name gooaq_pairs \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/gooaq_pairs/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/gooaq_pairs/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name yahoo_answers_title_answer \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/yahoo_answers_title_answer/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/yahoo_answers_title_answer/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name amazon_qa \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/amazon_qa/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/amazon_qa/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name agnews \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/agnews/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/agnews/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name npr \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/npr/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/npr/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name ccnews_title_text \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/ccnews_title_text/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/ccnews_title_text/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 1024

# python find_neg_pipeline.py \
#     --dataset_name zero_shot_re \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/zero_shot_re/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/zero_shot_re/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name xsum \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/xsum/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/xsum/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name amazon_review_2018 \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/amazon_review_2018/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/amazon_review_2018/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name s2orc_title_abstract \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/s2orc_title_abstract/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/s2orc_title_abstract/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name paq_pairs \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/paq_pairs/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/paq_pairs/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name medmcqa \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/medmcqa/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/medmcqa/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name wikihow \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/wikihow/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/wikihow/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name simple_wiki \
#     --retriever BAAI/bge-large-en-v1.5 \
#     --corpus_path fine_tuning/data/simple_wiki/collection/corpus.jsonl \
#     --top_k 200 \
#     --file_path fine_tuning/data/simple_wiki/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train \
#     --embedding_chunk_size 1 \
#     --query_chunk_size 2048

# python find_neg_pipeline.py \
#     --dataset_name squad \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_squad_index \
#     --top_k 200 \
#     --file_path fine_tuning/data/squad/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name tevatron_msmarco_psg \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_tevatron_msmarco_psg_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/tevatron_msmarco_psg/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name gooaq_pairs \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_gooaq_pairs_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/gooaq_pairs/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name yahoo_answers_title_answer \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_yahoo_answers_title_answer_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/yahoo_answers_title_answer/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name amazon_qa \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_amazon_qa_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/amazon_qa/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name agnews \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_agnews_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/agnews/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name npr \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_npr_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/npr/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name ccnews_title_text \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_ccnews_title_text_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/ccnews_title_text/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name zero_shot_re \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_zero_shot_re_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/zero_shot_re/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name xsum \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_xsum_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/xsum/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name medmcqa \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_medmcqa_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/medmcqa/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train
    
# python find_neg_pipeline.py \
#     --dataset_name wikihow \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_wikihow_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/wikihow/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name simple_wiki \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_simple_wiki_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/simple_wiki/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name amazon_review_2018 \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_amazon_review_2018_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/amazon_review_2018/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name s2orc_title_abstract \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_s2orc_title_abstract_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/s2orc_title_abstract/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

# python find_neg_pipeline.py \
#     --dataset_name paq_pairs \
#     --retriever bm25 \
#     --corpus_path indexes/bm25_paq_pairs_index \
#     --top_k 100 \
#     --file_path fine_tuning/data/paq_pairs/train_positive_pids.jsonl \
#     --save_dir fine_tuning/data/train

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
 python mine/V6/pseudo_psg_scoring/find_neg_pipeline.py \
     --dataset_name eli5 \
     --retriever BAAI/bge-large-en-v1.5 \
     --corpus_path all_retrieval_data_used/eli5/collection/corpus.jsonl \
     --top_k 200 \
     --file_path all_retrieval_data_used/eli5/train_positive_pids.jsonl \
     --save_dir downstream_data_with_hn_for_each_retriever/eli5

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python mine/V6/pseudo_psg_scoring/find_neg_pipeline.py \
#     --dataset_name eli5 \
#     --retriever bm25 \
#     --corpus_path all_retrieval_data_used/eli5/collection/corpus.jsonl \
#     --top_k 100 \
#     --file_path all_retrieval_data_used/eli5/train_positive_pids.jsonl \
#     --save_dir downstream_data_with_hn_for_each_retriever/eli5

task="quora_duplicate"
model_name_or_path="checkpoint_dir/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_2_25_psg_temp_07"
metadata_fp="all_retrieval_data_used/${task}/train_positive_pids.jsonl"
p_embed_dir="${model_name_or_path}/embeddings/doc/$task/7_shards"
q_embed_dir="${model_name_or_path}/embeddings/query/$task/7_shards"
output_fp="${model_name_or_path}/hn_metadata/${task}.jsonl"
mkdir -p "${model_name_or_path}/hn_metadata"
corpus_path="all_retrieval_data_used/${task}/collection/corpus.jsonl"

CUDA_VISIBLE_DEVICES=0 \
 python mine/V6/pseudo_psg_scoring/find_neg_pipeline_q_with_multi_ppsg.py \
     --top_k 100 \
     --metadata_fp $metadata_fp \
     --p_embed_dir $p_embed_dir \
     --q_embed_dir $q_embed_dir \
     --output_fp $output_fp \
     --corpus_path $corpus_path \
     --max_p_embedding_batch 800000 \
     --query_chunk_size 8192 \
     --use_cuda 1 \
     --extra_q_to_pos_fp "quora_q_to_poss_dict.json"