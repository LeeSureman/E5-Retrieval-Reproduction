python gen_synthetic_data_according_to_tasks.py \
  --api_model gpt-3.5-turbo-0613 \
  --input_dir synthetic_data_2024_2_19_prompt_task_2/gpt4_100 \
  --num_examples_each_task 1 \
  --output_suffix tmp_p3 \
  --task_type short-long \
  --example_num 100

task_type="long-long"
python gen_synthetic_data_according_to_tasks.py \
  --api_model gpt-3.5-turbo-0613 \
  --input_dir synthetic_data_2024_2_19_prompt_task_2/gpt3_0301 \
  --num_examples_each_task 1 \
  --output_suffix tmp_p3 \
  --task_type $task_type \
  --example_num 50

task_type="long-long"
python gen_synthetic_data_according_to_tasks.py \
  --api_model gpt-4 \
  --input_dir test_indomain_data/arguana \
  --num_examples_each_task 2 \
  --output_suffix repeat_2 \
  --task_type $task_type