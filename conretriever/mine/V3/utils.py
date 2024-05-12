import copy

from safetensors import safe_open
from safetensors.torch import save_file
import os
import tqdm
import json

def delete_one_prefix_in_ckpt_keys(ckpt_dir, prefix):

    fp_s = os.listdir(ckpt_dir)
    fp_s = list(filter(lambda x:x.endswith('.safetensors'),fp_s))

    for fp in tqdm.tqdm(fp_s):
        with safe_open(os.path.join(ckpt_dir, fp), framework="pt", device="cpu") as f:
            for key in f.keys():
                assert key.startswith(prefix)

    for fp in tqdm.tqdm(fp_s):
        true_fp = os.path.join(ckpt_dir, fp)
        ckpt_dict_new = {}
        with safe_open(true_fp, framework="pt", device="cpu") as f:
            for key in f.keys():
                key_new = key[len(prefix):]
                ckpt_dict_new[key_new] = f.get_tensor(key)


        save_file(ckpt_dict_new, true_fp, metadata={"format": "pt"})

    safetensors_index_js_fp = '{}/model.safetensors.index.json'.format(ckpt_dir)
    if os.path.exists(safetensors_index_js_fp):
        safetensors_index_js = json.load(open(safetensors_index_js_fp))
        safetensors_index_js_new = copy.deepcopy(safetensors_index_js)
        for k,v in safetensors_index_js['weight_map'].items():
            k_new = k[len(prefix):]
            safetensors_index_js_new['weight_map'][k_new] = v
            del safetensors_index_js_new['weight_map'][k]

        json.dump(safetensors_index_js_new, open(safetensors_index_js_fp,'w'))


def add_one_prefix_in_ckpt_keys(ckpt_dir, prefix):
    fp_s = os.listdir(ckpt_dir)
    fp_s = list(filter(lambda x:x.endswith('.safetensors'),fp_s))

    for fp in tqdm.tqdm(fp_s):
        true_fp = os.path.join(ckpt_dir, fp)
        ckpt_dict_new = {}
        with safe_open(true_fp, framework="pt", device="cpu") as f:
            for key in f.keys():
                key_new = prefix + key
                ckpt_dict_new[key_new] = f.get_tensor(key)

        save_file(ckpt_dict_new, true_fp, metadata={"format": "pt"})

    safetensors_index_js_fp = '{}/model.safetensors.index.json'.format(ckpt_dir)
    safetensors_index_js = json.load(open(safetensors_index_js_fp))
    safetensors_index_js_new = copy.deepcopy(safetensors_index_js)
    for k,v in safetensors_index_js['weight_map'].items():
        k_new = prefix + k
        safetensors_index_js_new['weight_map'][k_new] = v
        del safetensors_index_js_new[k]

    json.dump(safetensors_index_js_new, open(safetensors_index_js_fp,'w'))

