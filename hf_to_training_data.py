from datasets import load_dataset
import os, jsonlines

save_dir = 'training_data/reproduction'
os.makedirs(save_dir, exist_ok=True)

task_names = ['fever', 'triviaqa', 'synthetic', 'quora_duplicate', 'eli5', 'hotpot_qa', 'squad', 'fiqa', 'nq', 'msmarco_passage']
for task_name in task_names:
    task_data = load_dataset('BeastyZ/E5-R', task_name, split='train')
    saved_data = []
    for d in task_data:
        if task_name == 'synthetic':
            saved_data.append({
                'query': d['query'],
                'positive': d['positive'],
                'negative': d['negative'],
                'instruction': d['instruction']
            })
        else:
            saved_data.append({
                'query': d['query'],
                'positive': d['positive'],
                'negative': d['negative'],
            })

    save_path = os.path.join(save_dir, f'{task_name}.jsonl')
    with jsonlines.open(save_path, 'w') as f:
        f.write_all(saved_data)
