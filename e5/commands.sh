CUDA_VISIBLE_DEVICES=4,5 bash scripts/eval_mteb_beir.sh intfloat/e5-mistral-7b-instruct

ckpt_dir=/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_new
CUDA_VISIBLE_DEVICES=1,6 \
representation_token_num=16 \
OUTPUT_DIR="$ckpt_dir/mteb_evaluation" \
bash scripts/eval_mteb_beir.sh $ckpt_dir

for ckpt_dir in \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_new \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-200 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-400 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-600 \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-400
do
model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1,6 \
representation_token_num=16 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
bash scripts/eval_mteb_beir.sh $ckpt_dir

done

 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-200 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-400 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-600




for ckpt_dir in \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_new \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-200 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-400 \
/remote-home/xnli/mycode/fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5/checkpoint-600 \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-400
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
mkdir -p $OUTPUT_DIR

task_names="ArguAna FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval TRECCOVID Touche2020"

CUDA_VISIBLE_DEVICES=1,6 \
representation_token_num=16 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
bash scripts/eval_mteb_beir.sh $ckpt_dir

done

for ckpt_dir in \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay \
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
mkdir -p $OUTPUT_DIR

task_names="ArguAna FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="NQ"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
representation_token_num=16 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
bash scripts/eval_mteb_beir.sh $ckpt_dir

done

for ckpt_dir in \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-200 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-400 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-600
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
mkdir -p $OUTPUT_DIR

task_names="ArguAna FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="NQ"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
representation_token_num=16 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done

for ckpt_dir in \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-200 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-400 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-600
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
mkdir -p $OUTPUT_DIR

task_names="QuoraRetrieval TRECCOVID Touche2020"
#task_names="NQ"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
representation_token_num=16 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done





../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_use_weight_decay

/cpfs01/projects-HDD/cfff-6ef6b3b71ce2_HDD/public/models/e5_mistral_7b

../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_new

../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-200 \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-400 \
../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_2e_4_with_weight_decay/checkpoint-600


for ckpt_dir in \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_new
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="tmp_evaluation"
mkdir -p $OUTPUT_DIR

task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
representation_token_num=16 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done

rm tmp_evaluation/*


python show_evaluation_result.py --ckpt_fp_s ../../fast_chat/checkpoint_dir/msmarco_with_hn_use_lora_lr_1e_4_with_weight_decay


for ckpt_dir in \
../../tevatron/examples/repllama/model_repllama
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="tmp_evaluation"
mkdir -p $OUTPUT_DIR

task_names="ArguAna Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="HotpotQA MSMARCO FEVER NQ"
#task_names="NQ"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
representation_token_num=1 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=512 \
passage_max_length=512 \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done


../../../../public/models/repllama-v1-7b-lora-passage
/remote-home/share/models/repllama-v1-7b-lora-passage

cd ~/mycode/e5/V2/


../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_21/checkpoint-200
../../tevatron/examples/repllama/model_repllama_no_lora_2024_1_22/checkpoint-600
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp/checkpoint-200

checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_23_wrap_qp
model_repllama_no_lora_mistral
../../tevatron/examples/repllama/model_repllama_no_lora_mistral

../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp/checkpoint-200

../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v2_no_gc/checkpoint-200/ \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v3_gc/checkpoint-200/ \
../../tevatron/examples/repllama/model_repllama_no_lora_mistral_new/checkpoint-200

../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v6_gc_small_bs

../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v2_no_gc/checkpoint-600/ \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v3_gc/checkpoint-600/

checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v9_gc_large_bs_my_msmarco

checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v11_gc_large_bs_xueguang_data_no_hn/checkpoint-200
checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v10_gc_large_bs_my_msmarco_no_hn/checkpoint-200

checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v8_gc_large_bs_my_rep_token/checkpoint-200

for ckpt_dir in \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v11_gc_large_bs_xueguang_data_no_hn/checkpoint-200 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v10_gc_large_bs_my_msmarco_no_hn/checkpoint-200
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
#OUTPUT_DIR="tmp_evaluation"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
representation_token_num=1 \
representation_id=2 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=512 \
passage_max_length=512 \
wrap_q_p=1 \
force_retrieval_model="default" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done

checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn


cd ../e5/V2

../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn/checkpoint-100 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn/checkpoint-200 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn/checkpoint-300 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn/checkpoint-400 \
../../fast_chat/checkpoint_dir/msmarco_with_hn_no_lora_lr_1e_5_xueguang_recipe_1_at_2024_1_22_wrap_qp_v12_gc_large_bs_1_hn/checkpoint-500

for ckpt_dir in \
../../fast_chat/checkpoint_dir/multi_dataset_first_run/checkpoint-200 \
../../fast_chat/checkpoint_dir/multi_dataset_first_run/checkpoint-400 \
../../fast_chat/checkpoint_dir/multi_dataset_first_run/checkpoint-600 \
../../fast_chat/checkpoint_dir/multi_dataset_first_run/checkpoint-800
do

model_dtype=fp16
OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
#OUTPUT_DIR="tmp_evaluation"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
representation_token_num=1 \
representation_id=2 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=512 \
passage_max_length=512 \
wrap_q_p="instruction" \
force_retrieval_model="default" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done

cd ../e5/V3

../../fast_chat/checkpoint_dir/multi_dataset_1hn/checkpoint-100 \
../../fast_chat/checkpoint_dir/multi_dataset_1hn/checkpoint-200 \
../../fast_chat/checkpoint_dir/multi_dataset_1hn/checkpoint-300 \
../../fast_chat/checkpoint_dir/multi_dataset_1hn

for ckpt_dir in \
../../tevatron/examples/repllama/model_repllama_no_lora_mistral_new/checkpoint-400
do

model_dtype=fp16
#OUTPUT_DIR="$ckpt_dir/mteb_evaluation/$model_dtype"
OUTPUT_DIR="tmp_evaluation"
mkdir -p $OUTPUT_DIR

#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS"
#task_names="ClimateFEVER FEVER HotpotQA MSMARCO NQ"
#task_names="NQ"
task_names="ArguAna TRECCOVID Touche2020 FiQA2018 NFCorpus SciFact SCIDOCS QuoraRetrieval"
#task_names="ArguAna"

CUDA_VISIBLE_DEVICES=0,1,2 \
representation_token_num=1 \
representation_id=2 \
OUTPUT_DIR=$OUTPUT_DIR \
model_dtype=$model_dtype \
task_names=$task_names \
query_max_length=512 \
passage_max_length=512 \
wrap_q_p="query_or_passage" \
force_retrieval_model="default" \
bash scripts/eval_mteb_beir.sh $ckpt_dir | tee -a "$OUTPUT_DIR/evaluation.log"

done