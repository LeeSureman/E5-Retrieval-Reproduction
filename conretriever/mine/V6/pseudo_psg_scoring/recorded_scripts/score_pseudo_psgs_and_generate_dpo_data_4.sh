#task="quora_duplicate"
#task='fiqa'
for task in fiqa quora_duplicate
do
model_name_or_path="checkpoint_dir/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_2_25_psg_temp_07"
metadata_fp="all_retrieval_data_used/${task}/train_positive_pids.jsonl"
p_embed_dir="${model_name_or_path}/embeddings/doc/$task/7_shards"
q_embed_dir="${model_name_or_path}/embeddings/query/$task/7_shards"
hn_metadata_fp="${model_name_or_path}/hn_metadata/${task}.jsonl"

ignore_first_n_neg=0
min_positive_sim=0
rejected_pair_num=4
consider_no_psg=0

#output_fp="${model_name_or_path}/dpo_data/$task/ignore_neg_${ignore_first_n_neg}_min_pos_sim_${min_positive_sim}_rejected_${rejected_pair_num}_none_psg_${consider_no_psg}.json"
output_dir="${model_name_or_path}/dpo_data/$task/tmp4"
mkdir $output_dir -p

echo "output_dir=${output_dir}"

python mine/V6/pseudo_psg_scoring/score_ppsg_by_pos_neg.py \
  --p_embed_dir $p_embed_dir \
  --q_embed_dir $q_embed_dir \
  --hn_metadata_fp $hn_metadata_fp \
  --ignore_first_n_neg $ignore_first_n_neg \
  --min_positive_sim $min_positive_sim \
  --rejected_pair_num $rejected_pair_num \
  --task_name $task \
  --output_dir $output_dir \
  --consider_no_psg $consider_no_psg \
  --test_num 500

echo "output_dir=${output_dir}"

done