import argparse
import random

import torch
import os
import tqdm
import jsonlines
import torch.nn as nn
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_embed_dir', required=True)
    parser.add_argument('--q_embed_dir', required=True)
    parser.add_argument('--hn_metadata_fp', required=True)
    parser.add_argument('--ignore_first_n_neg', type=int, required=True,
                        help='ignore the first n negatives for mitigating false-negatives')
    # 5
    parser.add_argument('--min_positive_sim', type=float, required=True,
                        help='skip the pair whose positive sim is too small, mitigating false-negatives')
    # 0.1
    parser.add_argument('--rejected_pair_num', type=int, required=True)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--consider_no_psg',required=True, type=int)
    parser.add_argument('--test_num',type=int,required=True)
    parser.add_argument('--filter_rejected_higher_than_no_psg',type=int,required=True)

    args = parser.parse_args()

    assert args.ignore_first_n_neg >= 0

    hn_metadata = list(jsonlines.open(args.hn_metadata_fp))
    q_to_hn_metadata = {}
    for tmp_js in hn_metadata:
        q_to_hn_metadata[tmp_js['query']] = tmp_js

    p_embed_filename_s = os.listdir(args.p_embed_dir)
    p_embed_js_s = []
    for fname in p_embed_filename_s:
        fp = os.path.join(args.p_embed_dir, fname)
        js_s_shard = torch.load(open(fp, 'rb'))
        p_embed_js_s.extend(js_s_shard)
    p_embed_js_s.sort(key=lambda x: x['gidx'])

    p_embeddings = list(map(lambda x: x['embed'].unsqueeze(0), p_embed_js_s))
    p_embeddings = torch.cat(p_embeddings, dim=0)
    print('p_embeddings: {}'.format(p_embeddings.size()))

    q_embeds_filename_s = os.listdir(args.q_embed_dir)
    q_embeds_js_s = []
    for fname in q_embeds_filename_s:
        fp = os.path.join(args.q_embed_dir, fname)
        js_s_shard = torch.load(open(fp, 'rb'))
        q_embeds_js_s.extend(js_s_shard)
    q_embeds_js_s.sort(key=lambda x: x['gidx'])

    # q_embeds_js_s = q_embeds_js_s[:200]

    queries = list(map(lambda x: x['query'], q_embeds_js_s))

    num_embeds_each_q = len(q_embeds_js_s[0]['embeds'])

    q_embeddings = []
    for tmp_js in q_embeds_js_s:
        q_embeddings.extend(list(map(lambda x: x.unsqueeze(0), tmp_js['embeds'])))

    q_embeddings = torch.cat(q_embeddings, dim=0)
    print('q_embeddings: {}'.format(q_embeddings.size()))

    pid_to_embedding = {}
    for tmp_js in p_embed_js_s:
        pid = tmp_js['id']
        p_embed = tmp_js['embed']
        pid_to_embedding[pid] = p_embed

    no_psg_best_num = 0

    output_data = {}
    skip_num = 0
    for i, q_js in enumerate(tqdm.tqdm(q_embeds_js_s)):
        q = q_js['query']
        if q in output_data:
            print('repeated query, so continue')
            continue
        pos_id = q_to_hn_metadata[q]['positive_pids'][0]
        neg_id_s = q_to_hn_metadata[q]['negative_pids']
        pseudo_psgs = q_js['pseudo_psg']

        neg_id_s = neg_id_s[args.ignore_first_n_neg:]

        pos_embedding = pid_to_embedding[pos_id]
        neg_embeddings = []
        for neg_id in neg_id_s:
            neg_embeddings.append(pid_to_embedding[neg_id])

        pos_negs_embeddings = torch.stack([pos_embedding] + neg_embeddings).to(torch.float32)
        q_ppsgs_embeddings = torch.stack(q_js['embeds']).to(torch.float32)

        q_ppsgs_to_ps_similiraties = torch.matmul(q_ppsgs_embeddings, pos_negs_embeddings.T) * 100

        q_ppsgs_scores = nn.functional.softmax(q_ppsgs_to_ps_similiraties, dim=1)[:, 0].tolist()

        if max(q_ppsgs_scores) == q_ppsgs_scores[-1]:
            no_psg_best_num += 1

        if args.min_positive_sim > q_ppsgs_scores[-1]:
            skip_num += 1
            if skip_num % 1000 == 0:
                print('the {} th example pos score is too small ({} < {}), '
                      'so skip. now skip_num: {}'.format(i, q_ppsgs_scores[-1], args.min_positive_sim, skip_num))
            continue

        if args.consider_no_psg:
            pseudo_psgs.append('None.')
            raise NotImplementedError
        else:
            q_without_psg_score = q_ppsgs_scores[-1]
            q_ppsgs_scores=q_ppsgs_scores[:-1]


        ppsg_with_score_s = list(zip(pseudo_psgs, q_ppsgs_scores))

        ppsg_with_score_s.sort(key=lambda x: x[-1], reverse=True)

        tmp_js_output = {}
        scores = []
        pairs = []
        responses = []
        sft_target = ppsg_with_score_s[0][0]

        for j in range(args.rejected_pair_num):
            n_responses = len(responses)
            chosen_response = ppsg_with_score_s[0][0]
            rejected_response = ppsg_with_score_s[-(j + 1)][0]
            chosen_score = ppsg_with_score_s[0][1]
            rejected_score = ppsg_with_score_s[-(j + 1)][1]

            if args.filter_rejected_higher_than_no_psg and rejected_score >= q_without_psg_score:
                break

            responses.append(chosen_response)
            responses.append(rejected_response)
            scores.append(chosen_score)
            scores.append(rejected_score)

            pairs.append((n_responses, n_responses + 1))

        if len(pairs) == 0:
            continue

        tmp_js_output['pairs'] = pairs
        tmp_js_output['responses'] = responses
        tmp_js_output['scores'] = scores
        tmp_js_output['task_name'] = args.task_name
        tmp_js_output['sft_target'] = sft_target

        output_data[q] = tmp_js_output

    print('input size: {}'.format(len(queries)))
    print('skip num: {}'.format(skip_num))

    if args.test_num > 0:
        print('split {} examples into test set'.format(args.test_num))
        output_data_k_v_list = list(output_data.items())
        random.shuffle(output_data_k_v_list)
        test_output_data_list = output_data_k_v_list[:args.test_num]
        train_output_data_list = output_data_k_v_list[args.test_num:]

        train_output_data = {}
        for k, v in train_output_data_list:
            train_output_data[k] = v

        test_output_data = {}
        for k, v in test_output_data_list:
            test_output_data[k] = v

        train_output_fp='{}/train.json'.format(args.output_dir)
        test_output_fp='{}/test.json'.format(args.output_dir)

        print('train output_data size: {}'.format(len(train_output_data)))

        json.dump(train_output_data, open(train_output_fp, 'w'))
        json.dump(test_output_data, open(test_output_fp, 'w'))

        pairs_num=0
        for q, v in train_output_data.items():
            pairs_num+=len(v['pairs'])
        print('pairs_num: {}, pairs/prompts: {}'.format(pairs_num, pairs_num/(len(train_output_data))))



    else:
        print('split {} examples into test set'.format(0))
        train_output_fp='{}/train.json'.format(args.output_dir)
        train_output_data = output_data

        print('train output_data size: {}'.format(len(output_data)))
        json.dump(train_output_data, open(train_output_fp, 'w'))

        pairs_num=0
        for q, v in train_output_data.items():
            pairs_num+=len(v['pairs'])
        print('pairs_num: {}, pairs/prompts: {}'.format(pairs_num, pairs_num/(len(train_output_data))))
