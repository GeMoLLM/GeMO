import argparse
from tqdm import tqdm
import numpy as np
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, default='copydetect_scores_gen')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
args = parser.parse_args()

file_paths = []
input_filename = f'./{args.input_filename}'

with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

all_scores = []
for file_path in tqdm(file_paths):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    filename = f'scores_{fileid}_{pid}.npy'
    
    scores = np.load(os.path.join(args.input_folder, filename))
    print(f"{fileid}_{pid}: {scores.mean()}")
    all_scores.append(scores)
    
all_scores = np.array(all_scores)
np.save(osp.join(args.input_folder, 'all_scores.npy'), all_scores)