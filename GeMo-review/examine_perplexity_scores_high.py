import argparse
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n', type=int, default=10)
args = parser.parse_args()

folder_path = 'perplexity_scores'

result = np.zeros((args.n, 750))
for i in range(args.n):
    data_path = f'perplexity_{args.prefix}-{i}_merged.npy'
    result[i] = np.load(osp.join(folder_path, data_path))
                
out_path = osp.join(folder_path, f'perplexity_{args.prefix}.npy')
print(out_path)
np.save(out_path, result)