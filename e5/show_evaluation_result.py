import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_fp_s',)

    args = parser.parse_args()

    ckpt_fp_s = args.ckpt_fp_s.split(' ')

    ckpt_to_eval_result = {}

    for ckpt_fp in ckpt_fp_s:
        ckpt_to_eval_result[ckpt_fp] = {}
        eval_dir = '{}/mteb_evaluation/fp16'.format(ckpt_fp)
        if os.path.exists(eval_dir):

            for data_eval_js_file_name in os.listdir(eval_dir):
                if not data_eval_js_file_name.endswith('.json'):
                    continue
                data_name = data_eval_js_file_name.split('.')[0]
                data_eval_fp = os.path.join(eval_dir, data_eval_js_file_name)
                print('data_eval_fp: {}'.format(data_eval_fp))
                data_eval_js = json.load(open(data_eval_fp))

                ckpt_to_eval_result[ckpt_fp][data_name] = data_eval_js['test']['ndcg_at_10']

    for ckpt_fp in ckpt_to_eval_result:
        print(ckpt_fp)
        for k,v in sorted(ckpt_to_eval_result[ckpt_fp].items()):
            print('{}: {}'.format(k,v))
        # print(ckpt_to_eval_result[ckpt_fp])
