import argparse
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
i1_list = []
i2_list = []
for fileid, pid, idx, typ in files:
    info_file = f'{fileid}_problem_info_{pid}.json'
    prob_name = json.load(open(info_file))['name']
    if typ == 'train':
        code_path = f'{fileid}_{pid}_solutions_{idx}.txt'
        desc1_path = f'parsed_description_gpt-3.5-instruct_{fileid}_{pid}_solution_{idx}.json'
        alg1_path = f'parsed_alg_ds_gpt-3.5-instruct_{fileid}_{pid}_solution_{idx}.json'
        
        desc2_path = f'parsed_description_gpt-4_{fileid}_{pid}_solution_{idx}.json'
        alg2_path = f'parsed_alg_ds_gpt-4_{fileid}_{pid}_solution_{idx}.json'
        
        assert osp.exists(code_path), f'code_path={code_path}'
        assert osp.exists(desc1_path), f'desc_path={desc1_path}'
        assert osp.exists(alg1_path), f'alg_path={alg1_path}'
        assert osp.exists(desc2_path), f'desc_path={desc2_path}'
        assert osp.exists(alg2_path), f'alg_path={alg2_path}'
        
        subfolder = osp.join(new_folder, f'{fileid}_{pid}_{idx}_src')
        
    elif typ == 'gpt4':
        code_path = f'{fileid}_solution_codeonly_{idx}_{pid}.py'
        desc1_path = f'parsed_description_gpt-3.5-instruct_{fileid}_solution_codeonly_{idx}_{pid}.json'
        alg1_path = f'parsed_alg_ds_gpt-3.5-instruct_{fileid}_solution_codeonly_{idx}_{pid}.json'
        
        desc2_path = f'parsed_description_gpt-4_{fileid}_solution_codeonly_{idx}_{pid}.json'
        alg2_path = f'parsed_alg_ds_gpt-4_{fileid}_solution_codeonly_{idx}_{pid}.json'

        assert osp.exists(code_path), f'code_path={code_path}'
        assert osp.exists(desc1_path), f'desc_path={desc1_path}'
        assert osp.exists(alg1_path), f'alg_path={alg1_path}'
        assert osp.exists(desc2_path), f'desc_path={desc2_path}'
        assert osp.exists(alg2_path), f'alg_path={alg2_path}'

        subfolder = osp.join(new_folder, f'{fileid}_{pid}_{idx}_gen')
    
    os.makedirs(subfolder, exist_ok=True)
    
    with open(osp.join(subfolder, 'problem_name.txt'), 'w') as f:
        f.write(prob_name)
    
    i1 = np.random.randint(2)
    i2 = 1 - i1
    i1_list.append(i1)
    i2_list.append(i2)
    shutil.copy(code_path, osp.join(subfolder, code_path))
    shutil.copy(desc1_path, osp.join(subfolder, desc1_path.replace('gpt-3.5-instruct', str(i1))))
    shutil.copy(alg1_path, osp.join(subfolder, alg1_path.replace('gpt-3.5-instruct', str(i1))))

    shutil.copy(desc2_path, osp.join(subfolder, desc2_path.replace('gpt-4', str(i2))))
    shutil.copy(alg2_path, osp.join(subfolder, alg2_path.replace('gpt-4', str(i2))))
    
np.savez('sample_for_annotation.npz', i1=i1_list, i2=i2_list)