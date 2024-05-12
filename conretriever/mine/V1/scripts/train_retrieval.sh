echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "master_port=$master_port"
echo "nproc_per_node=$nproc_per_node"

echo ""
echo ""

echo "n_hard_negative=$n_hard_negative"
echo "per_device_train_batch_size=$per_device_train_batch_size"
echo "learning_rate=$learning_rate"

echo ""
echo ""

echo "model_name_or_path=$model_name_or_path"
echo "data_path=$data_path"

echo ""
echo ""

echo "representation_token_num=$representation_token_num"
echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
echo "logging_dir=$logging_dir"

echo "************************************"
echo "************************************"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port ./mine/train_retrieval.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
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
    --representation_token_num $representation_token_num \
    --remove_unused_columns False \
    --chunk_sizes 16 \
    --dataloader_num_workers 4 \
    --logging_dir $logging_dir