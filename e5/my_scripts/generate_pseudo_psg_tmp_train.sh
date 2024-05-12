#sft_template="mistral"
#model_name_or_path="/remote-home/share/models/mistral_7b_instruct"
sft_template="vicuna"
model_name_or_path="/remote-home/xnli/mycode/fast_chat/sft_ckpt/slim_orca_full"
temperature=0.7

export CUDA_VISIBLE_DEVICES=0

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS NQ"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="MSMARCO"
#task_names="ArguAna"
#tasks="HotpotQA"
#tasks="hotpot_qa"
#tasks="fever hotpot_qa nq"
tasks="fever hotpot_qa nq msmarco_passage"
#tasks="FEVER HotpotQA NQ MSMARCO"

#tasks="SciFact NQ MSMARCO"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"

query_dir="/remote-home/xnli/mycode/fast_chat/nq_msmarco_fever_hotpotqa_data_dir"

model_dtype="float16"
# model_name_or_path="../../fast_chat/sft_ckpt/merged_light"
pseudo_passage_output_dir="/remote-home/xnli/mycode/fast_chat/train_pseudo_psg/silm_orca/temperature_$temperature"
mkdir -p $pseudo_passage_output_dir

python generate_pseudo_psg.py \
  --model_name_or_path $model_name_or_path \
  --max_new_tokens 512 \
  --query_dir $query_dir\
  --tasks "$tasks" \
  --temperature $temperature \
  --output_dir $pseudo_passage_output_dir \
  --retrieval_query_max_length 512 \
  --model_dtype $model_dtype \
  --sft_template $sft_template \
  --max_num_seqs 256