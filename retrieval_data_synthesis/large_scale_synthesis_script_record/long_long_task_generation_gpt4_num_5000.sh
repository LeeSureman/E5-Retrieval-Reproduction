task_type="long-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 250 \
    --api_model gpt-4 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/gpt4/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_2

task_type="long-long"
python gen_synthetic_data_according_to_tasks.py \
  --api_model gpt-3.5-turbo-0613 \
  --input_dir "synthetic_data_2024_2_19_prompt_task_2/gpt4" \
  --num_examples_each_task 2 \
  --output_suffix repeat_2 \
  --task_type $task_type