#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="tmp-outputs/"
fi

python e5/fix_ckpt_keys.py --ckpt_dir "${MODEL_NAME_OR_PATH}"

python -u e5/mteb_beir_eval.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --output-dir "${OUTPUT_DIR}" "$@" \
    --representation_token_num $representation_token_num \
    --representation_id $representation_id \
    --model_dtype $model_dtype \
    --task_names "$task_names" \
    --query_max_length $query_max_length \
    --passage_max_length $passage_max_length \
    --wrap_q_p $wrap_q_p \
    --force_retrieval_model $force_retrieval_model \
    --batch_per_gpu $batch_per_gpu \
    --pseudo_passage_fp $pseudo_passage_fp \
    --how_to_use_pseudo_passage $how_to_use_pseudo_passage \
    --doc_embedding_cache_dir $doc_embedding_cache_dir

echo "done"
