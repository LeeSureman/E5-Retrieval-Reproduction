model_name_or_path="/remote-home/share/models/mistral_7b_base"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nproc_per_node=8
per_device_train_batch_size=16
gradient_accumulation_steps=4

weight_decay="0."
lr_scheduler_type="linear"
warmup_ratio="0"
warmup_steps="30"

output_dir="sft_ckpt/share_gpt34_full"
mkdir -p $output_dir
echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
my_logging_dir="$output_dir/my_logs.txt"
echo "logging_dir=$logging_dir"
echo "my_logging_dir=$my_logging_dir"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node=8 --master_port=20001 mine/V6/train_sft.py \
    --model_name_or_path $model_name_or_path \
    --data_path sft_data/share_gpt34_full.jsonl \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 600 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay $weight_decay \
    --warmup_ratio $warmup_ratio \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --logging_steps 1 \
    --fsdp "shard_grad_op auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True \
    --model_max_length 1280 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    | tee -a $my_logging_dir