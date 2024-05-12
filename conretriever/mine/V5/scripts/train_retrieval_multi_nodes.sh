export NCCL_DEBUG=INFO
#output_dir="checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5"
output_dir="checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4"
use_lora="1"
learning_rate="1e-4"
do_grad_cache="1"
per_device_train_batch_size=512
n_hard_negative=1

query_max_length=128
doc_max_length=384

echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
echo "logging_dir=$logging_dir"
echo "use_lora=$use_lora"

echo "************************************"
echo "************************************"

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=20001 --master_addr 172.20.73.220 \
./mine/V3/train_retrieval.py \
    --model_name_or_path /cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base \
    --data_path /cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/lxn_20110240012/dataset/with_hn/tmp_msmarco_with_hn_bm25_bge.jsonl \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 100 \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "shard_grad_op auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True \
    --report_to tensorboard \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --n_hard_negative $n_hard_negative \
    --representation_token_num 16 \
    --remove_unused_columns False \
    --chunk_sizes 32 \
    --dataloader_num_workers 4 \
    --logging_dir $logging_dir \
    --use_lora $use_lora \
    --do_grad_cache $do_grad_cache \
    --query_max_length $query_max_length \
    --doc_max_length $doc_max_length