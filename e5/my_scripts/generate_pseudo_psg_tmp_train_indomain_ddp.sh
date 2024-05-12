#sft_template="mistral"
#model_name_or_path="/remote-home/share/models/mistral_7b_instruct"
sft_template="vicuna"
model_name_or_path="/remote-home/xnli/mycode/fast_chat/sft_ckpt/slim_orca_full"

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export nproc_per_node=8
#total_gpu=$nproc_per_node

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
                                          ##task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS NQ"
                                          ##task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="MSMARCO"
#task_names="ArguAna"
#tasks="HotpotQA"
#tasks="hotpot_qa"
#tasks="fever hotpot_qa nq"
tasks="fever hotpot_qa nq msmarco_passage"
#tasks="fever"
temperature=1.0
sample_n=4
#tasks="SciFact NQ MSMARCO"

#query_dir="/remote-home/xnli/mycode/fast_chat/nq_msmarco_fever_hotpotqa_data_dir"
#query_dir="/remote-home/xnli/mycode/fast_chat/nq_msmarco_fever_hotpotqa_data_dir"
query_dir="/remote-home/xnli/mycode/fast_chat/nq_msmarco_fever_hotpotqa_data_dir"

model_dtype="float16"
# model_name_or_path="../../fast_chat/sft_ckpt/merged_light"
#pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/train_pseudo_psg/silm_orca/ddp_try_tmp_3090_124_temperature_$temperature"
pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/train_pseudo_psg/silm_orca/sample_${sample_n}_temperature_$temperature"
mkdir -p $pseudo_passage_output_dir

total_gpu=8

pid_list=""

for shard_index in `seq 0 "$((total_gpu-1))"`
do
  echo $shard_index
#  CUDA_VISIBLE_DEVICES=$shard_index

#  if [ "$shard_index" == "0" ]
  CUDA_VISIBLE_DEVICES=$shard_index \
  python generate_pseudo_psg_ddp.py \
    --model_name_or_path $model_name_or_path \
    --max_new_tokens 512 \
    --query_dir $query_dir\
    --tasks "$tasks" \
    --temperature $temperature \
    --output_dir $pseudo_passage_output_dir \
    --retrieval_query_max_length 512 \
    --model_dtype $model_dtype \
    --sft_template $sft_template \
    --max_num_seqs 128 \
    --shard_index $shard_index \
    --total_gpu $total_gpu \
    --sample_n $sample_n \
    &

  pid_list="${pid_list} $!"
done

echo "all process have been started, wait for all of them finishing"
echo "all_pids: ${pid_list}"

for tmp_pid in $pid_list
do
  wait $tmp_pid
done


#CUDA_VISIBLE_DEVICES=0
#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#python generate_pseudo_psg_ddp.py \
#  --model_name_or_path $model_name_or_path \
#  --max_new_tokens 512 \
#  --query_dir $query_dir\
#  --tasks "$tasks" \
#  --temperature $temperature \
#  --output_dir $pseudo_passage_output_dir \
#  --retrieval_query_max_length 512 \
#  --model_dtype $model_dtype \
#  --sft_template $sft_template \
#  --max_num_seqs 256 \
#  --shard_index $CUDA_VISIBLE_DEVICES \
#  --total_gpu $total_gpu
#
#CUDA_VISIBLE_DEVICES=1
#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#python generate_pseudo_psg_ddp.py \
#  --model_name_or_path $model_name_or_path \
#  --max_new_tokens 512 \
#  --query_dir $query_dir\
#  --tasks "$tasks" \
#  --temperature $temperature \
#  --output_dir $pseudo_passage_output_dir \
#  --retrieval_query_max_length 512 \
#  --model_dtype $model_dtype \
#  --sft_template $sft_template \
#  --max_num_seqs 256 \
#  --shard_index $CUDA_VISIBLE_DEVICES \
#  --total_gpu $total_gpu


python generate_pseudo_psg_ddp_merge.py \
  --model_name_or_path $model_name_or_path \
  --max_new_tokens 512 \
  --query_dir $query_dir\
  --tasks "$tasks" \
  --temperature $temperature \
  --output_dir $pseudo_passage_output_dir \
  --retrieval_query_max_length 512 \
  --model_dtype $model_dtype \
  --sft_template $sft_template \
  --max_num_seqs 256 \
  --total_gpu $total_gpu