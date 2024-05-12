sft_template="mistral"
model_name_or_path="/remote-home/share/models/mistral_7b_instruct"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS NQ"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="MSMARCO"
#task_names="ArguAna"
tasks="FEVER"
#tasks="SciFact NQ MSMARCO"

query_dir="beir_test_queries"

# model_name_or_path="../../fast_chat/sft_ckpt/merged_light"
for tasks in MSMARCO NQ FEVER HotpotQA
do
for prompt_index in 0 1 2 3 4 5 6
do
  pseudo_passage_output_dir="mistral_instruct_pseudo_psgs_by_prompt_$prompt_index/"
  mkdir -p $pseudo_passage_output_dir

#  python generate_pseudo_psg.py \
#    --model_name_or_path $model_name_or_path \
#    --max_new_tokens 512 \
#    --query_dir $query_dir\
#    --tasks "$tasks" \
#    --temperature 0. \
#    --output_dir $pseudo_passage_output_dir \
#    --retrieval_query_max_length 512 \
#    --model_dtype float16 \
#    --sft_template $sft_template \
#    --use_prompt_candidate_template $prompt_index


  #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  # model_name_or_path="../../fast_chat/sft_ckpt/merged_light"
  #pseudo_psg_dir="$model_name_or_path/pseudo_psgs"

  for ckpt_dir in \
  BAAI/bge-large-en-v1.5
  do

  model_dtype=fp16
  #OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
  OUTPUT_DIR="$pseudo_passage_output_dir/$tasks/mteb_evaluation"
#  OUTPUT_DIR="tmp_${task_names}_$prompt_index"
  mkdir -p $OUTPUT_DIR

  #task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
  #task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS NQ"
  #task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
  #task_names="NQ"
  #task_names="MSMARCO"
  #task_names="ArguAna"
  task_names=$tasks

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  representation_token_num=1 \
  representation_id=2 \
  OUTPUT_DIR=$OUTPUT_DIR \
  model_dtype=$model_dtype \
  task_names=$task_names \
  query_max_length=512 \
  passage_max_length=512 \
  wrap_q_p="instruction" \
  force_retrieval_model="default" \
  batch_per_gpu=128 \
  pseudo_passage_fp=$pseudo_passage_output_dir \
  how_to_use_pseudo_passage="embedding_average" \
  doc_embedding_cache_dir="../embedding_caches/bge_large" \
  bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

  done

done
done
123