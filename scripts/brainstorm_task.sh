task_type="short-long"
python synthesize/brainstorm_retrieval_task.py \
    --brainstorm_num 1 \
    --api_model gpt-3.5-turbo \
    --save_dir synthesize/synthetic_data \
    --task_type $task_type \
    --sample_prompt_tasks_num 4 \
    --sample_prompt_tasks_from synthesize/seed_tasks

task_type="long-long"
python synthesize/brainstorm_retrieval_task.py \
    --brainstorm_num 1 \
    --api_model gpt-3.5-turbo  \
    --save_dir synthesize/synthetic_data \
    --task_type $task_type \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from synthesize/seed_tasks
