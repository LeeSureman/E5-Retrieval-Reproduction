task_type="short-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 3000 \
    --api_model gpt-3.5-turbo-0301 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/gpt3_given_a_retrieve/$task_type" \
    --sample_prompt_tasks_num 4 \
    --sample_prompt_tasks_from seed_tasks_starts_with_given_a_retrieve_format

task_type="short-long"
python gen_synthetic_data_according_to_tasks.py \
  --api_model gpt-3.5-turbo-0613 \
  --input_dir "synthetic_data_2024_2_19_prompt_task_2/gpt3_given_a_retrieve" \
  --num_examples_each_task 3 \
  --output_suffix repeat_3 \
  --task_type $task_type