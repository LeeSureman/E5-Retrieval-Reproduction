#sft_template="mistral"
#model_name_or_path="/remote-home/share/models/mistral_7b_instruct"
sft_template="vicuna"
model_name_or_path="/remote-home/xnli/mycode/fast_chat/sft_ckpt/slim_orca_full"

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export nproc_per_node=8
#total_gpu=$nproc_per_node



#tasks="HotpotQA"
#tasks="hotpot_qa"
#tasks="fever hotpot_qa nq"
#tasks="fever hotpot_qa nq msmarco_passage"
#tasks="nq fever msmarco_passage hotpot_qa eli5 squad triviaqa quora_duplicate"

#max_model_len=1024
#tasks="msmarco_passage"
#tasks="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval ClimateFEVER FEVER HotpotQA MSMARCO NQ DBPedia"
#tasks="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS FEVER HotpotQA MSMARCO NQ"
#cqadup_tasks="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"
#tasks="DBPedia QuoraRetrieval ClimateFEVER $cqadup_tasks"
#tasks="$cqadup_tasks"
#tasks="ArguAna"


#tasks="fiqa fever hotpot_qa triviaqa quora_duplicate synthetic"
#tasks="synthetic"
#max_model_len=1536

#tasks="msmarco_passage nq squad eli5"
tasks="quora_duplicate"
max_model_len=1024


temperature=1.5
sample_n=32

model_dtype="float16"

query_dir="/remote-home/xnli/mycode/fast_chat/retrieval_data_for_run/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_quora_light"

#pseudo_passage_output_dir="pseudo_psg_for_test/tmp_debug"
pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/pseudo_psg_for_run/msmarco_hybrid_plus_e5_hybrid_plus_synthetic/slim_orca_sample_${sample_n}_temp_$temperature"
mkdir -p $pseudo_passage_output_dir



gpu_index_array=(0 1 2 3 4 5 6 7)
total_gpu="${#gpu_index_array[@]}"
echo "total_gpu=$total_gpu"

pid_list=""

#for shard_index in `seq 0 "$((total_gpu-1))"`
#do
#
#  CUDA_VISIBLE_DEVICES="${gpu_index_array[${shard_index}]}"
#
#  echo "start shard ${shard_index} on GPU ${CUDA_VISIBLE_DEVICES}"
#
#  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#  python generate_pseudo_psg_ddp_v2.py \
#    --model_name_or_path $model_name_or_path \
#    --max_new_tokens 512 \
#    --query_dir $query_dir\
#    --tasks "$tasks" \
#    --temperature $temperature \
#    --output_dir $pseudo_passage_output_dir \
#    --retrieval_query_max_length 512 \
#    --model_dtype $model_dtype \
#    --sft_template $sft_template \
#    --max_num_seqs 96 \
#    --shard_index $shard_index \
#    --total_gpu $total_gpu \
#    --sample_n $sample_n \
#    --max_model_len $max_model_len \
#    --gpu_memory_utilization 0.9 \
#    &
#
#  pid_list="${pid_list} $!"
#done
#
#echo "all process have been started, wait for all of them finishing"
#echo "all_pids: ${pid_list}"
#
#for tmp_pid in $pid_list
#do
#  wait $tmp_pid
#done



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
  --max_num_seqs 32 \
  --total_gpu $total_gpu