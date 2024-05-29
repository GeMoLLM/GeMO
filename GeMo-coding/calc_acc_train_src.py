import numpy as np
import json
input_filename = './codeforces_A_file_paths_final.txt'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

acc_list = []
for file_path in file_paths:
    d = json.load(open(file_path))
    n_cor = d['n_solution_py3']
    n_inc = d['n_incorrect_solution_py3']
    acc = n_cor / (n_cor + n_inc)
    acc_list.append(acc)
print(acc_list)
np.save('acc_train_src.npy', np.array(acc_list))
