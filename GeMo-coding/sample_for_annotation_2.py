import numpy as np
import os
import os.path as osp
import random
import shutil
import json

input_filename = './codeforces_A_file_paths_final.txt'
train_idx_filename = 'codeforces_A_train_index.npy'
gpt4_idx_filename = 'codeforces_A_gen_gpt4_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)
gpt4_idx = np.load(gpt4_idx_filename)

all_files = []
for file, train_indices, gpt4_indices in zip(file_paths, train_idx, gpt4_idx):
    fileid = file.split('_')[0]
    pid = file.split('_')[-1].split('.')[0]
    for idx in train_indices:
        all_files.append((fileid, pid, idx, 'train'))
    for idx in gpt4_indices:
        all_files.append((fileid, pid, idx, 'gpt4'))

assert len(all_files) == 4000, f'len(all_files)={len(all_files)}'

files = random.sample(all_files, 40)

new_folder = 'sample_for_annotation'
os.makedirs(new_folder, exist_ok=True)

for fileid, pid, idx, typ in files:
    info_file = f'{fileid}_problem_info_{pid}.json'
    prob_name = json.load(open(info_file))['name']
    if typ == 'train':
        code_path = f'{fileid}_{pid}_solutions_{idx}.txt'
        desc_path = f'parsed_description_gpt-3.5-instruct_{fileid}_{pid}_solution_{idx}.json'
        alg_path = f'parsed_alg_ds_gpt-3.5-instruct_{fileid}_{pid}_solution_{idx}.json'
        assert osp.exists(code_path), f'code_path={code_path}'
        assert osp.exists(desc_path), f'desc_path={desc_path}'
        assert osp.exists(alg_path), f'alg_path={alg_path}'
        subfolder = osp.join(new_folder, f'{fileid}_{pid}_src')
        
    elif typ == 'gpt4':
        code_path = f'{fileid}_solution_codeonly_{idx}_{pid}.py'
        desc_path = f'parsed_description_gpt-3.5-instruct_{fileid}_solution_codeonly_{idx}_{pid}.json'
        alg_path = f'parsed_alg_ds_gpt-3.5-instruct_{fileid}_solution_codeonly_{idx}_{pid}.json'
        assert osp.exists(code_path), f'code_path={code_path}'
        assert osp.exists(desc_path), f'desc_path={desc_path}'
        assert osp.exists(alg_path), f'alg_path={alg_path}'
        subfolder = osp.join(new_folder, f'{fileid}_{pid}_gen')
    
    os.makedirs(subfolder, exist_ok=True)
    
    with open(osp.join(subfolder, 'problem_name.txt'), 'w') as f:
        f.write(prob_name)
    
    shutil.copy(code_path, osp.join(subfolder, code_path.split('/')[-1]))
    shutil.copy(desc_path, osp.join(subfolder, desc_path.split('/')[-1]))
    shutil.copy(alg_path, osp.join(subfolder, alg_path.split('/')[-1]))