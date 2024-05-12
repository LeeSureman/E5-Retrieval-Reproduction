# self synthetic data with no neg doc.

output_dir="checkpoint_dir/self_syn_0422_1"
use_lora="1"
learning_rate="5e-5"
temperature="100"
do_grad_cache="1"
per_device_train_batch_size=84
n_hard_negative=0
num_train_epochs=1
batch_same_task="1"
flash_attention="1"
seed=300

representation_id=2
representation_token_num=1
wrap_q_p="instruction"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
nproc_per_node=6

model_name_or_path="mistralai/Mistral-7B-v0.1"
# data_root_dir="./tevatron_original_data_light"
data_root_dir="/home/paibo/model_retriever/syn_data_self"
# 这个文件夹里面的msmarco_doc是bm25的hard negative

# data_path="/remote-home/xnli/mycode/msmarco_data/with_hn/tmp_msmarco_with_hn_xueguang.jsonl"

# model_name_or_path="/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/mistral_7b_base"
# data_root_dir="./multi_retrieval_data_small"
# data_path="/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/lxn_20110240012/dataset/with_hn/tmp_msmarco_with_hn_xueguang.jsonl"

weight_decay="0"
lr_scheduler_type="linear"
warmup_ratio="0"
warmup_steps="30"

# weight_decay="0.1"
# weight_decay=0
# lr_scheduler_type="linear"
# warmup_ratio="0"
# warmup_steps="100"

query_max_length=512
doc_max_length=512

mkdir -p $output_dir
echo "output_dir=$output_dir"
logging_dir="$output_dir/transformers_logs"
my_logging_dir="$output_dir/my_logs.txt"
echo "logging_dir=$logging_dir"
echo "my_logging_dir=$my_logging_dir"

echo "************************************"
echo "************************************"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
torchrun --nproc_per_node=$nproc_per_node --master_port=42001 ./mine/V7/train_retrieval.py \
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
    --logging_dir $logging_dir \
    --use_lora $use_lora \
    --do_grad_cache $do_grad_cache \
    --query_max_length $query_max_length \
    --doc_max_length $doc_max_length \
    --wrap_q_p $wrap_q_p \
    --data_sample_config task_to_sample_weight_1 \
    --temperature $temperature \
    --batch_same_task $batch_same_task \
    --flash_attention $flash_attention \
    --seed $seed \
    --lora_r 16 \
    | tee -a $my_logging_dir

# python mine/V7/fix_ckpt_keys.py --ckpt_dir $output_dir

cd ../e5/V6

    # print(task_names)
    # ['ArguAna', 'ClimateFEVER', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval',
    #  'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval',
    #  'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval',
    #  'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval',
    #  'CQADupstackWordpressRetrieval', 'DBPedia', 'FEVER', 'FiQA2018', 'HotpotQA', 'MSMARCO', 'NFCorpus', 'NQ',
    #  'QuoraRetrieval', 'SCIDOCS', 'SciFact', 'Touche2020', 'TRECCOVID']


for ckpt_dir in \
/home/paibo/model_retriever/fast_chat/checkpoint_dir/self_syn_0422_1
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
#OUTPUT_DIR="tmp_evaluation_2"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval FEVER HotpotQA MSMARCO NQ ClimateFEVER DBPedia"
#task_names="ArguAna"
small_task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
cqadup_task_names="CQADupstackAndroidRetrieval CQADupstackEnglishRetrieval CQADupstackGamingRetrieval CQADupstackGisRetrieval CQADupstackMathematicaRetrieval CQADupstackPhysicsRetrieval CQADupstackProgrammersRetrieval CQADupstackStatsRetrieval CQADupstackTexRetrieval CQADupstackUnixRetrieval CQADupstackWebmastersRetrieval CQADupstackWordpressRetrieval"
large_task_names="NQ DBPedia HotpotQA FEVER ClimateFEVER MSMARCO"
task_names="$small_task_names $cqadup_task_names $large_task_names"
#task_names="$small_task_names $cqadup_task_names"

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
batch_per_gpu=128 \
pseudo_passage_fp="0" \
how_to_use_pseudo_passage="concat" \
doc_embedding_cache_dir="None" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done
