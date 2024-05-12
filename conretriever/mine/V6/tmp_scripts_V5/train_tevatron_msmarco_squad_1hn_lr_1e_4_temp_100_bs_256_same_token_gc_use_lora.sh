#output_dir="checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn"
#output_dir="checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay"
output_dir="checkpoint_dir/tevatron_msmarco_squad_lr_1e_4_temp_100_bs_256_same_token_gc_use_lora"
use_lora="1"
learning_rate="1e-4"
temperature="100"
do_grad_cache="1"
per_device_train_batch_size=64
n_hard_negative=1
num_train_epochs=1

representation_id=2
representation_token_num=1
wrap_q_p="query_or_passage"

CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4

model_name_or_path="/remote-home/share/models/mistral_7b_base"
data_root_dir="./tevatron_original_data_light_msmarco_squad"

#data_path="/remote-home/xnli/mycode/msmarco_data/with_hn/tmp_msmarco_with_hn_xueguang.jsonl"

#model_name_or_path="/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base"
#data_root_dir="./multi_retrieval_data_small"
#data_path="/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/lxn_20110240012/dataset/with_hn/tmp_msmarco_with_hn_xueguang.jsonl"

weight_decay="0"
lr_scheduler_type="linear"
warmup_ratio="0"
warmup_steps="30"

#weight_decay="0.1"
#weight_decay=0
#lr_scheduler_type="linear"
#warmup_ratio="0"
#warmup_steps="100"

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

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node=$nproc_per_node --master_port=43001 ./mine/V5/train_retrieval.py \
    --model_name_or_path $model_name_or_path \
    --data_root_dir $data_root_dir \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 100 \
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
    --chunk_sizes 16 \
    --dataloader_num_workers 4 \
    --logging_dir $logging_dir \
    --use_lora $use_lora \
    --do_grad_cache $do_grad_cache \
    --query_max_length $query_max_length \
    --doc_max_length $doc_max_length \
    --wrap_q_p $wrap_q_p \
    --data_sample_config task_to_sample_weight_1 \
    --temperature $temperature \
    | tee -a $my_logging_dir

#python mine/V3/fix_ckpt_keys.py --ckpt_dir $output_dir

cd ../e5/V4

    # print(task_names)
    # ['ArguAna', 'ClimateFEVER', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval',
    #  'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval',
    #  'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval',
    #  'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval',
    #  'CQADupstackWordpressRetrieval', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ',
    #  'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID']


for ckpt_dir in \
../../fast_chat/checkpoint_dir/tevatron_msmarco_squad_lr_1e_4_temp_100_bs_256_same_token_gc_use_lora/checkpoint-600 \
../../fast_chat/checkpoint_dir/tevatron_msmarco_squad_lr_1e_4_temp_100_bs_256_same_token_gc_use_lora/checkpoint-1200 \
../../fast_chat/checkpoint_dir/tevatron_msmarco_squad_lr_1e_4_temp_100_bs_256_same_token_gc_use_lora
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
#OUTPUT_DIR="tmp_evaluation_2"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
representation_token_num=1 \
representation_id=2 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=512 \
passage_max_length=512 \
wrap_q_p="query_or_passage" \
force_retrieval_model="default" \
batch_per_gpu=16 \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done