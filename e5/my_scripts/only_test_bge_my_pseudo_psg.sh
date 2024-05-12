CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#pseudo_psg_dir="$model_name_or_path/pseudo_psgs"
pseudo_psg_dir="pseudo_psg_for_test/slim_orca_sft"
doc_embedding_cache_dir="../embedding_caches/bge_large"

for ckpt_dir in \
BAAI/bge-large-en-v1.5
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="$pseudo_psg_dir/mteb_evaluation"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS NQ"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="MSMARCO"
#task_names="ArguAna"

#tasks="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval ClimateFEVER FEVER HotpotQA MSMARCO NQ DBPedia"
#cqadup_tasks="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"

small_task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
cqadup_task_names="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"
large_task_names="NQ DBPedia HotpotQA FEVER ClimateFEVER MSMARCO"
task_names="$small_task_names $cqadup_task_names $large_task_names"



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
batch_per_gpu=64 \
pseudo_passage_fp="$pseudo_psg_dir" \
doc_embedding_cache_dir=$doc_embedding_cache_dir \
how_to_use_pseudo_passage='embedding_average' \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done