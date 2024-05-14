task_type="short-long"
python synthesize/gen_synthetic_data.py \
  --api_model gpt-3.5-turbo \
  --input_path synthesize/synthetic_data/brainstorm/gpt-3.5-turbo_short-long_syn_data.jsonl \
  --num_examples_each_task 1 \
  --task_type $task_type


task_type="long-long"
python synthesize/gen_synthetic_data.py \
  --api_model gpt-3.5-turbo \
  --input_path synthesize/synthetic_data/brainstorm/gpt-3.5-turbo_long-long_syn_data.jsonl \
  --num_examples_each_task 1 \
  --task_type $task_type

