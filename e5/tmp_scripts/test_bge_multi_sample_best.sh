CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for task_names in HotpotQA
do
for ckpt_dir in \
BAAI/bge-large-en-v1.5
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="bge_evaluation_multi_sample_best"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="FEVER"
#task_names="HotpotQA"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
representation_token_num=1
representation_id=2
OUTPUT_DIR=$OUTPUT_DIR
model_dtype=$model_dtype
task_names=$task_names
query_max_length=512
passage_max_length=512
wrap_q_p="instruction"
force_retrieval_model="default"
batch_per_gpu=64
pseudo_passage_fp="mistral_instruct_pseudo_psg_multi_sample_temperature_1"
how_to_use_pseudo_passage="embedding_average"
doc_embedding_cache_dir="../embedding_caches/bge_large"

python -u mteb_beir_eval_each_query_multi_sample_best.py \
    --model-name-or-path $ckpt_dir \
    --output-dir "${OUTPUT_DIR}" \
    --representation_token_num $representation_token_num \
    --representation_id $representation_id \
    --model_dtype $model_dtype \
    --task_names "$task_names" \
    --query_max_length $query_max_length \
    --passage_max_length $passage_max_length \
    --wrap_q_p $wrap_q_p \
    --force_retrieval_model $force_retrieval_model \
    --batch_per_gpu $batch_per_gpu \
    --pseudo_passage_fp $pseudo_passage_fp \
    --how_to_use_pseudo_passage $how_to_use_pseudo_passage \
    --doc_embedding_cache_dir $doc_embedding_cache_dir \
    --need_scores_each_q 1

done

done

rm $OUTPUT_DIR -r