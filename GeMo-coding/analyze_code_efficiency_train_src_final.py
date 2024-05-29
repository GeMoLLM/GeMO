import numpy as np
import os.path as osp

input_filename = './codeforces_A_file_paths_final.txt'
train_idx_filename = 'codeforces_A_train_index.npy'

n_probs = 100
n_src = 20

folder = 'run_judge_stats'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

runtimes = np.zeros((n_probs, n_src, 10), dtype=int)
memories = np.zeros((n_probs, n_src, 10), dtype=int)

for fi, (file_path, indices) in enumerate(zip(file_paths, train_idx)):

    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    for i, idx in enumerate(indices):
        filename = osp.join(folder, f'src_stats_{fileid}_{pid}_{idx}.txt')
        cur_runtimes = []
        cur_memories = []
        with open(filename) as f:
            for line in f:
                parts = line.split()
                cur_runtimes.append(int(float(parts[5])))
                cur_memories.append(int(parts[8]))
                
        runtimes[fi][i] = cur_runtimes
        memories[fi][i] = cur_memories
    
np.savez(osp.join(folder, f'src_stats_final.npz'), runtimes=runtimes, memories=memories)