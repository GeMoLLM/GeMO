import argparse
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser(description='Parse run judge output for generated code.')
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--n_gen', type=int, default=29)
args = parser.parse_args()

input_filename = './codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

folder = 'judge_output/'
n_correct_mat = np.zeros((args.n_gen, len(file_paths)), dtype=int)
for i in range(args.n_gen):
    for j, file_path in enumerate(file_paths):
        fileid = file_path.split('_')[0]
        fileidno = fileid.split('-')[-1]
        pid = file_path.split('_')[-1].split('.')[0]
        filename = f'{fileid}_solution_{args.input_id}_{i}_{pid}.py'
        with open(osp.join(folder, filename), 'r') as f:
            lines = f.readlines()
            compiled = int(lines[0].strip())
            n_cor = int(lines[1].strip())
            runtime = int(lines[2].strip())
            n_correct_mat[i][j] = n_cor

n_acc_mat = np.where(n_correct_mat == 10, 1, 0)
print(n_acc_mat.shape)
    
n_acc_mat_agg = n_acc_mat.sum(axis=0)

print(np.sum(n_acc_mat_agg < 5))