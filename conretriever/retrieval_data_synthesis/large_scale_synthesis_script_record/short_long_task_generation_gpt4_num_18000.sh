task_type="short-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 900 \
    --api_model gpt-4 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/gpt4_2/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_2

#这个生成的任务不是都从given a开头的，已经把所有的given a开头的任务放到synthetic_data_2024_2_19_prompt_task_2/gpt4_given_a里了