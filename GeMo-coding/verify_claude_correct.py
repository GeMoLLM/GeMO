import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='claude_codeonly_temp-1.0_p-0.9')
parser.add_argument('--idx_fileid', type=str, default='temp-1.0_p-0.9')
args = parser.parse_args()

input_filename = './codeforces_A_file_paths_claude_final.txt'
train_idx_filename = f'codeforces_A_gen_claude_{args.idx_fileid}_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 50, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

for file_path, indices in zip(file_paths, train_idx):
    if args.input_id == 'claude_codeonly_temp-1.0_p-0.9':
        if file_path in [
            'train-00002_problem_info_14.json',
        ]:
            continue
    if args.input_id == 'claude_codeonly_temp-0.5_p-0.9':
        if file_path in [
            'train-00002_problem_info_14.json',
            'train-00006_problem_info_4.json',
            'train-00008_problem_info_21.json',
            'train-00012_problem_info_25.json'
        ]:
            continue
    # the only exception
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    for idx in indices:
        filename = f'run_judge_stats/stats_{fileid}_{pid}_gen_{args.input_id}_{idx}.txt'
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                assert 'passed' in line, f'line={line}, filename={filename}'