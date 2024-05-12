import sys
sys.path.append('..')
sys.path.append('.')
from utils import delete_one_prefix_in_ckpt_keys

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir',required=True)

    args = parser.parse_args()

    delete_one_prefix_in_ckpt_keys(args.ckpt_dir, 'model.')