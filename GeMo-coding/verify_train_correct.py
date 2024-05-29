import numpy as np

input_filename = './codeforces_A_file_paths_final.txt'
train_idx_filename = 'codeforces_A_train_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

for file_path, indices in zip(file_paths, train_idx):
    # the only exception
    if file_path == 'train-00005_problem_info_13.json':
        continue

    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    for idx in indices:
        filename = f'run_judge_stats/src_stats_{fileid}_{pid}_{idx}.txt'
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                assert 'passed' in line, f'line={line}, filename={filename}'