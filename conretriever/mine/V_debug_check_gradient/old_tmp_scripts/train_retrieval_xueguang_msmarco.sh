output_dir="checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_hn_xueguang"
#output_dir="checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay"
use_lora="0"
learning_rate="1e-5"
do_grad_cache="1"
per_device_train_batch_size=32
n_hard_negative=7
num_train_epochs=1

model_name_or_path="/remote-home/share/models/mistral_7b_base"
data_path="/remote-home/xnli/mycode/msmarco_data/with_hn/tmp_msmarco_with_hn_xueguang.jsonl"

#weight_decay="0.1"
#lr_scheduler_type="linear"
#warmup_ratio="0"
#warmup_steps="100"

#weight_decay="0.1"
weight_decay=0
lr_scheduler_type="linear"
warmup_ratio="0"
warmup_steps="100"

query_max_length=128
doc_max_length=384

mkdir -p $output_dir
echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
my_logging_dir="$output_dir/my_logs.txt"
echo "logging_dir=$logging_dir"
echo "my_logging_dir=$my_logging_dir"

echo "************************************"
echo "************************************"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=20001 ./mine/V3/train_retrieval.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
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
    --representation_token_num 1 \
    --remove_unused_columns False \
    --chunk_sizes 32 \
    --dataloader_num_workers 4 \
    --logging_dir $logging_dir \
    --use_lora $use_lora \
    --do_grad_cache $do_grad_cache \
    --query_max_length $query_max_length \
    --doc_max_length $doc_max_length \
    | tee -a $my_logging_dir