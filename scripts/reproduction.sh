# Lora args
use_lora=True
lora_r=16

# Model args
model_name_or_path=mistralai/Mistral-7B-v0.1
trust_remote_code=False
representation_id=2
representation_token_num=1
wrap_q_p=instruction

# Data args
data_root_dir=training_data/reproduction

# Training args
batch_same_task=True
flash_attention=True
do_grad_cache=True
query_max_length=512
doc_max_length=512
n_hard_negative=1
temperature=100

# Other args
CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4
learning_rate="5e-5"
per_device_train_batch_size=128
num_train_epochs=1
weight_decay="0"
lr_scheduler_type="linear"
warmup_ratio="0"
warmup_steps="30"
output_dir=conretriever/checkpoint_dir/reproduction

mkdir -p $output_dir
logging_dir=""
logging_file="$output_dir/my_logs.txt"

echo "output_dir=$output_dir"
echo "logging_file=$logging_file"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node=$nproc_per_node --master_port=42001 conretriever/train_retriever.py \
    --model_name_or_path $model_name_or_path \
    --data_root_dir $data_root_dir \
    --bf16 True \
    --trust_remote_code $trust_remote_code \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --warmup_ratio $warmup_ratio \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --logging_steps 1 \
    --fsdp "shard_grad_op auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True \
    --report_to tensorboard \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --n_hard_negative $n_hard_negative \
    --representation_id $representation_id \
    --representation_token_num $representation_token_num \
    --remove_unused_columns False \
    --chunk_sizes 64 \
    --dataloader_num_workers 4 \
    --logging_dir $output_dir/transformers_logs \
    --use_lora $use_lora \
    --lora_r $lora_r \
    --do_grad_cache $do_grad_cache \
    --query_max_length $query_max_length \
    --doc_max_length $doc_max_length \
    --wrap_q_p $wrap_q_p \
    --temperature $temperature \
    --batch_same_task $batch_same_task \
    --flash_attention $flash_attention \
    --seed 300 \
    | tee -a $logging_file


ckpt_dir=$output_dir
model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
mkdir -p $OUTPUT_DIR

small_task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
cqadup_task_names="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"
large_task_names="NQ DBPedia HotpotQA FEVER ClimateFEVER MSMARCO"
task_names="$small_task_names $cqadup_task_names $large_task_names"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
representation_token_num=$representation_token_num \
representation_id=$representation_id \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=512 \
passage_max_length=512 \
wrap_q_p="instruction" \
force_retrieval_model="default" \
batch_per_gpu=128 \
pseudo_passage_fp="0" \
how_to_use_pseudo_passage="concat" \
doc_embedding_cache_dir="None" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"
