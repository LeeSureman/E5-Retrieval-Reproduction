CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


for ckpt_dir in \
BAAI/bge-large-en-v1.5
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="bge_evaluation_pseudo_psg"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
task_names="MSMARCO"
#task_names="ArguAna"

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
batch_per_gpu=256 \
pseudo_passage_fp="mistral_instrcut_pseudo_psg_test" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done


123