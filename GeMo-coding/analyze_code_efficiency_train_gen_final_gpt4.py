import numpy as np
import os.path as osp

import argparse
parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
parser.add_argument('--model_fam', type=str, default='gpt4')
parser.add_argument('--idx_fileid', type=str, default='')
parser.add_argument('--n_probs', type=int, default=100)
args = parser.parse_args()

input_filename = f'./{args.input_filename}'
train_idx_filename = f'codeforces_A_gen_{args.model_fam}{args.idx_fileid}_index.npy'

n_probs = args.n_probs
n_gen = 20

folder = 'run_judge_stats'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

runtimes = np.zeros((n_probs, n_gen, 10), dtype=int)
memories = np.zeros((n_probs, n_gen, 10), dtype=int)

for fi, (file_path, indices) in enumerate(zip(file_paths, train_idx)):

    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    for i, idx in enumerate(indices):
        filename = osp.join(folder, f'stats_{fileid}_{pid}_gen_{args.input_id}_{idx}.txt')
        cur_runtimes = []
        cur_memories = []
        with open(filename) as f:
            for line in f:
                if 'timed out' in line:
                    cur_runtimes.append(60000)
                    cur_memories.append(100000)
                else:
                    parts = line.split()
                    cur_runtimes.append(int(float(parts[5])))
                    cur_memories.append(int(parts[8]))
                
        runtimes[fi][i] = cur_runtimes
        memories[fi][i] = cur_memories

out_file = osp.join(folder, f'stats_gen_final_{args.input_id}.npz')
print(out_file)
np.savez(out_file, runtimes=runtimes, memories=memories)