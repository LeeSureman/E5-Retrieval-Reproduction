model_name_or_path="checkpoint_dir/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_2_25_psg_temp_07"
use_flash_attention="0"

task='quora_duplicate'
input_fp="all_retrieval_data_used/$task/collection/corpus.jsonl"
encode_query_or_doc="doc"


gpu_index_array=(1 2 3 4 5 6 7)
total_gpu="${#gpu_index_array[@]}"
echo "total_gpu=$total_gpu"

embedding_output_dir="${model_name_or_path}/embeddings/${encode_query_or_doc}/$task/${total_gpu}_shards"
mkdir -p embedding_output_dir

pid_list=""

for shard_index in `seq 0 "$((total_gpu-1))"`
do

  CUDA_VISIBLE_DEVICES="${gpu_index_array[${shard_index}]}"

  echo "start shard ${shard_index} on GPU ${CUDA_VISIBLE_DEVICES}"

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  python mine/V6/pseudo_psg_scoring/encode_q_or_d_pool_ddp.py \
  --model_name_or_path $model_name_or_path \
  --input_fp $input_fp \
  --encode_query_or_doc $encode_query_or_doc \
  --representation_id 2 \
  --representation_token_num 1 \
  --max_length 384 \
  --batch_size 16 \
  --use_flash_attention $use_flash_attention \
  --shard_index $shard_index \
  --total_gpu $total_gpu \
  --sort_length_when_encoding 1 \
  --output_dir $embedding_output_dir \
  --pre_tokenize_all_input 1 \
  &

  pid_list="${pid_list} $!"
done

echo "all process have been started, wait for all of them finishing"
echo "all_pids: ${pid_list}"
echo "kill ${pid_list}"

for tmp_pid in $pid_list
do
  wait $tmp_pid
done