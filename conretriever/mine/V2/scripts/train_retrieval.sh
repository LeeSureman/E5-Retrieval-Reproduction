output_dir="checkpoint_dir/tmp"
use_lora="1"


echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
echo "logging_dir=$logging_dir"
echo "use_lora=$use_lora"

echo "************************************"
echo "************************************"

CUDA_VISIBLE_DEVICES=4,5 \
torchrun --nproc_per_node=2 --master_port=20001 ./mine/V2/train_retrieval.py \
    --model_name_or_path /remote-home/share/models/mistral_7b_base \
    --data_path /remote-home/xnli/mycode/msmarco_data/msmarco_train.jsonl \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
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
    --n_hard_negative 0 \
    --representation_token_num 16 \
    --remove_unused_columns False \
    --chunk_sizes 16 \
    --dataloader_num_workers 4 \
    --logging_dir $logging_dir \
    --use_lora $use_lora


output_dir="checkpoint_dir/tmp"
use_lora="1"


echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
echo "logging_dir=$logging_dir"
echo "use_lora=$use_lora"

echo "************************************"
echo "************************************"

CUDA_VISIBLE_DEVICES=4,5 \
torchrun --nproc_per_node=2 --master_port=20001 ./mine/V2/train_retrieval.py \
    --model_name_or_path /remote-home/share/models/mistral_7b_base \
    --data_path /remote-home/xnli/mycode/msmarco_data/msmarco_train_small.jsonl \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
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
    --n_hard_negative 0 \
    --representation_token_num 16 \
    --remove_unused_columns False \
    --chunk_sizes 16 \
    --dataloader_num_workers 4 \
    --logging_dir $logging_dir \
    --use_lora $use_lora