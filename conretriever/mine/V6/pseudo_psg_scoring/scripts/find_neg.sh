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