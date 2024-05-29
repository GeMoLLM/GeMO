import argparse
import numpy as np
from itertools import combinations
import json
import re
from utils import entropy, extract_complexity
from scipy.stats import mannwhitneyu


parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
args = parser.parse_args()

input_filename = './codeforces_A_file_paths_final.txt'
train_idx_filename = 'codeforces_A_train_index.npy'
train_gen_idx_filename = 'codeforces_A_gen_gpt4_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)
train_gen_idx = np.load(train_gen_idx_filename)

def compare(src, tgt):
    src_ori = src
    tgt_ori = tgt
    src = [x for x in src if x != -1]
    tgt = [x for x in tgt if x != -1]
    # print(src, tgt)
    stat, p_value = mannwhitneyu(src, tgt, alternative='less')
    if p_value < 0.05:
        return 1
    stat, p_value = mannwhitneyu(tgt, src, alternative='less')
    if p_value < 0.05:
        print(src, tgt, src_ori, tgt_ori)
        return -1
    return 0

time_complexity_cmp = []
space_complexity_cmp = []

l = []
for file_path, indices_src, indices_gen in zip(file_paths, train_idx, train_gen_idx):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    time_complexity_gen_list = []
    space_complexity_gen_list = []
    time_complexity_src_list = []
    space_complexity_src_list = []
    for si, gi in zip(indices_src, indices_gen):
        input_file_gen = f'parsed_description_{args.model}_{fileid}_solution_{args.input_id}_{gi}_{pid}.json'
        input_file_src = f'parsed_description_{args.model}_{fileid}_{pid}_solution_{si}.json'
        data_gen = json.load(open(input_file_gen))
        data_src = json.load(open(input_file_src))
        time_complexity_gen = extract_complexity(data_gen['time_complexity'])
        space_complexity_gen = extract_complexity(data_gen['space_complexity'])
        
        time_complexity_src = extract_complexity(data_src['time_complexity'])
        space_complexity_src = extract_complexity(data_src['space_complexity'])
        
        time_complexity_gen_list.append(time_complexity_gen)
        space_complexity_gen_list.append(space_complexity_gen)

        time_complexity_src_list.append(time_complexity_src)
        space_complexity_src_list.append(space_complexity_src)
        
    l += time_complexity_gen_list + space_complexity_gen_list + time_complexity_src_list + space_complexity_src_list
    # print(np.unique(time_complexity_gen_list))
    # print(np.unique(space_complexity_gen_list))
    # print(np.unique(time_complexity_src_list))
    # print(np.unique(space_complexity_src_list))
# print(np.unique(l))
    cmp_time_complexity = compare(time_complexity_src_list, time_complexity_gen_list)
    cmp_space_complexity = compare(space_complexity_src_list, space_complexity_gen_list)
    
    time_complexity_cmp.append(cmp_time_complexity)
    space_complexity_cmp.append(cmp_space_complexity)
    print(f"cmp: {cmp_time_complexity}, {cmp_space_complexity} of {fileid} {pid}")

out_file = f'complexity_train_solution_cmp_final_src_gen_{args.model}_{args.input_id}.npz'
print(out_file)
np.savez(out_file, time_complexity_cmp=time_complexity_cmp, space_complexity_cmp=space_complexity_cmp)