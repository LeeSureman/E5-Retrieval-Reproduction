#export NCCL_P2P_DISABLE="1"
#export NCCL_IB_DISABLE="1"
cd /remote-home/xnli/mycode/fast_chat
#model_name_or_path="/remote-home/share/models/mistral_7b_base"
task="fiqa"
retriever_ckpt_dir="checkpoint_dir/msmarco_hybrid_plus_e5_hybrid_plus_synthetic_2_25_psg_temp_07"
#dpo_data_path="${retriever_ckpt_dir}/dpo_data/$task/ignore_neg_0_min_pos_sim_0_rejected_3_none_psg_0.json"
dpo_data_path="${retriever_ckpt_dir}/dpo_data/fiqa/tmp5/train.json"
eval_dpo_data_path="${retriever_ckpt_dir}/dpo_data/fiqa/tmp5/test.json"
model_name_or_path="sft_ckpt/slim_orca_full"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nproc_per_node=8
#CUDA_VISIBLE_DEVICES=0,1
#nproc_per_node=2
per_device_train_batch_size=16
gradient_accumulation_steps=1

sft_learning_rate=1e-5
dpo_learning_rate=5e-7
num_sft_epochs=3
num_dpo_epochs=3

dpo_beta=0.01
sft_coef_when_dpo=0


weight_decay="0."
lr_scheduler_type="linear"
warmup_ratio="0"
warmup_steps="30"

model_max_length=768
max_prompt_length=192

train_mode="only_dpo"

if [ $train_mode == "sft_dpo" ]; then
  echo "sft+dpo training"
sft_output_dir="${model_name_or_path}/${task}_further_sft_4"
dpo_output_dir="$sft_output_dir/further_dpo"
elif [ $train_mode == "only_dpo" ]; then
  echo "only_dpo training"
  sft_output_dir="$model_name_or_path"
  dpo_output_dir="$sft_output_dir/${task}_further_dpo_9"

fi

#output_dir="${model_name_or_path}/further_dpo"

mkdir -p $sft_output_dir
mkdir -p $dpo_output_dir

echo "sft_output_dir=$sft_output_dir"
echo "dpo_output_dir=$dpo_output_dir"

my_sft_logging_dir="$sft_output_dir/my_logs.txt"
echo "my_sft_logging_dir=$my_sft_logging_dir"

my_dpo_logging_dir="$dpo_output_dir/my_logs.txt"
echo "my_dpo_logging_dir=$my_dpo_logging_dir"


if [ $train_mode == "sft_dpo" ]; then


  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  torchrun --nproc_per_node=$nproc_per_node --master_port=42001 mine/V6/train_dpo.py \
      --model_name_or_path $model_name_or_path \
      --dpo_data_path $dpo_data_path \
      --bf16 True \
      --output_dir $sft_output_dir \
      --num_train_epochs $num_sft_epochs \
      --per_device_train_batch_size $per_device_train_batch_size \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps $gradient_accumulation_steps \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 600 \
      --save_total_limit 100 \
      --learning_rate $sft_learning_rate \
      --weight_decay $weight_decay \
      --warmup_ratio $warmup_ratio \
      --warmup_steps $warmup_steps \
      --lr_scheduler_type $lr_scheduler_type \
      --logging_steps 1 \
      --fsdp "shard_grad_op auto_wrap" \
      --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
      --tf32 True \
      --model_max_length $model_max_length \
      --max_prompt_length $max_prompt_length \
      --gradient_checkpointing True \
      --lazy_preprocess True \
      --dpo_beta $dpo_beta \
      --sft_coef_when_dpo $sft_coef_when_dpo \
      --remove_unused_columns False \
      --flash_attention 1 \
      --dataloader_num_workers 2 \
      --sft_mode True \
      | tee -a $my_sft_logging_dir

fi


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node=$nproc_per_node --master_port=42001 mine/V6/train_dpo.py \
    --model_name_or_path $sft_output_dir \
    --dpo_data_path $dpo_data_path \
    --bf16 True \
    --output_dir $dpo_output_dir \
    --num_train_epochs $num_dpo_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 600 \
    --save_total_limit 100 \
    --learning_rate $dpo_learning_rate \
    --weight_decay $weight_decay \
    --warmup_ratio $warmup_ratio \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --logging_steps 1 \
    --fsdp "shard_grad_op auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True \
    --model_max_length $model_max_length \
    --max_prompt_length $max_prompt_length \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dpo_beta $dpo_beta \
    --sft_coef_when_dpo $sft_coef_when_dpo \
    --remove_unused_columns False \
    --flash_attention 1 \
    --dataloader_num_workers 2 \
    --sft_mode False \
    --eval_dpo_data_path $eval_dpo_data_path \
    --max_grad_norm 10
    | tee -a $my_dpo_logging_dir

echo "final_output_dir=$dpo_output_dir"

