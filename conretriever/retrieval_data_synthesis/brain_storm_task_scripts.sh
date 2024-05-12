# python gen_synthetic_data.py \
#     --brainstorm_num 5 \
#     --api_model xwin-70b-v0.1 \
#     --save_dir synthetic_data \
#     --overwrite

#python gen_synthetic_data.py \
#    --brainstorm_num 1 \
#    --api_model gpt-3.5-turbo-0125 \
#    --save_dir synthetic_data \
#    --overwrite

#task_type="short-long"
#python brainstorm_retrieval_task.py \
#    --brainstorm_num 5 \
#    --api_model gpt-4-0125-preview \
#    --save_dir synthetic_data_3 \
#    --task_type $task_type \
#    --specify_output_dir "synthetic_data_2024_2_18_p4/tmp_try_gpt4_0125_preview/$task_type"

task_type="short-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 5 \
    --api_model gpt-4 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_18_p14_top_1/tmp_try_gpt4/$task_type" \
    --sample_prompt_tasks_num 4 \
    --sample_prompt_tasks_from seed_tasks_2


task_type="short-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 10 \
    --api_model gpt-4 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/try_gpt4/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_2

task_type="short-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 10 \
    --api_model gpt-4 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/tmp_try_gpt4/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_2

task_type="short-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 10 \
    --api_model gpt-3.5-turbo-0301 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/tmp_try_gpt3_0613/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_starts_with_given_a

task_type="long-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 5 \
    --api_model gpt-3.5-turbo-0301 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/tmp_try_gpt3_0301_p2/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_fixed

task_type="long-long"
python brainstorm_retrieval_task.py \
    --brainstorm_num 10 \
    --api_model gpt-4 \
    --save_dir synthetic_data_3 \
    --task_type $task_type \
    --specify_output_dir "synthetic_data_2024_2_19_prompt_task_2/tmp_try_gpt4_p3/$task_type" \
    --sample_prompt_tasks_num 2 \
    --sample_prompt_tasks_from seed_tasks_fixed



gpt-4-0125-preview
gpt-4-1106-preview
gpt-4
123123
