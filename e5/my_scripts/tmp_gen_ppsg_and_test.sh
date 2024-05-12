#sft_template="mistral"
#model_name_or_path="/remote-home/share/models/mistral_7b_instruct"
sft_template="vicuna"
#model_name_or_path="/remote-home/xnli/mycode/fast_chat/sft_ckpt/slim_orca_full"
model_name_or_path="/remote-home/xnli/mycode/say_i_donot_know/.cache/xnli/lxn_quora_9_2024-03-12_18-40-47_490949/step-BEST/hf_ckpt"
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
tasks="QuoraRetrieval"

max_model_len=2048
#tasks="eli5 fiqa synthetic"
#tasks="eli5"


#max_model_len=1024
#tasks="quora_duplicate nq fever"
#tasks="fever"

#max_model_len=1024
#tasks="triviaqa hotpot_qa squad"

#tasks="msmarco_passage"
#tasks="fiqa"
#tasks="synthetic"
#tasks="fever"

#temperature=0.7
#sample_n=1

temperature=0
sample_n=1

model_dtype="float16"
#temperature=1.0
#sample_n=4

#tasks="SciFact NQ MSMARCO"

#query_dir="/remote-home/xnli/mycode/fast_chat/nq_msmarco_fever_hotpotqa_data_dir"
#query_dir="/remote-home/xnli/mycode/fast_chat/nq_msmarco_fever_hotpotqa_data_dir"
#query_dir="/remote-home/xnli/mycode/fast_chat/retrieval_data_for_run/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_quora_light"
query_dir="beir_test_queries"
#pseudo_passage_output_dir="pseudo_psg_for_test/tmp_debug"
pseudo_passage_output_dir="${model_name_or_path}/pseudo_psgs"
mkdir -p $pseudo_passage_output_dir

# model_name_or_path="../../fast_chat/sft_ckpt/merged_light"
#pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/train_pseudo_psg/silm_orca/ddp_try_tmp_3090_124_temperature_$temperature"
#pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/train_pseudo_psg/silm_orca/sample_${sample_n}_temperature_$temperature"

#pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/pseudo_psg_for_run/msmarco_hybrid_plus_e5_hybrid_plus_synthetic/slim_orca_sample_${sample_n}_temp_$temperature"
#pseudo_passage_output_dir=tmp_eli5_psg_2

gpu_index_array=(0 1)
total_gpu="${#gpu_index_array[@]}"
echo "total_gpu=$total_gpu"

pid_list=""

for shard_index in `seq 0 "$((total_gpu-1))"`
do

  CUDA_VISIBLE_DEVICES="${gpu_index_array[${shard_index}]}"

  echo "start shard ${shard_index} on GPU ${CUDA_VISIBLE_DEVICES}"

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  python generate_pseudo_psg_ddp_v2.py \
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
    --max_model_len $max_model_len \
    &

  pid_list="${pid_list} $!"
done

echo "all process have been started, wait for all of them finishing"
echo "all_pids: ${pid_list}"

for tmp_pid in $pid_list
do
  wait $tmp_pid
done



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

export CUDA_VISIBLE_DEVICES=0,1
pseudo_passage_fp=$pseudo_passage_output_dir
for ckpt_dir in \
../../fast_chat/checkpoint_dir/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_2_25_psg_temp_07
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="$pseudo_passage_fp/mteb_evaluation"
#OUTPUT_DIR="tmp_evaluation_2"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval FEVER HotpotQA MSMARCO NQ ClimateFEVER DBPedia"
#task_names="ArguAna"
small_task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
cqadup_task_names="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"
large_task_names="NQ DBPedia HotpotQA FEVER ClimateFEVER MSMARCO"
task_names="$small_task_names $cqadup_task_names $large_task_names"
#task_names="$small_task_names"
task_names=$tasks

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
representation_token_num=1 \
representation_id=2 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=1024 \
passage_max_length=512 \
wrap_q_p="instruction" \
force_retrieval_model="default" \
batch_per_gpu=32 \
pseudo_passage_fp=$pseudo_passage_fp \
how_to_use_pseudo_passage="concat" \
doc_embedding_cache_dir="$ckpt_dir/test_doc_cache" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done