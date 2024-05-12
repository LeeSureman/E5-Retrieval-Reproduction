sft_template="mistral"
model_name_or_path="/remote-home/share/models/mistral_7b_instruct"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS NQ"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="MSMARCO"
#task_names="ArguAna"
#tasks="HotpotQA"
#tasks="hotpot_qa"
#tasks="fever hotpot_qa nq"
tasks="NQ FEVER HotpotQA MSMARCO"
#tasks="SciFact NQ MSMARCO"

query_dir="beir_test_queries"

model_dtype="float16"
# model_name_or_path="../../fast_chat/sft_ckpt/merged_light"
pseudo_passage_output_dir="mistral_instruct_pseudo_psg"
mkdir -p $pseudo_passage_output_dir

python generate_pseudo_psg.py \
  --model_name_or_path $model_name_or_path \
  --max_new_tokens 512 \
  --query_dir $query_dir\
  --tasks "$tasks" \
  --temperature 0. \
  --output_dir $pseudo_passage_output_dir \
  --retrieval_query_max_length 512 \
  --model_dtype $model_dtype \
  --sft_template $sft_template \
  --max_num_seqs 256