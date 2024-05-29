import argparse
import numpy as np
from itertools import combinations
import json
import re
from utils import entropy, extract_complexity

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--model_fam', type=str, default='gpt4')
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
parser.add_argument('--idx_fileid', type=str, default='')
args = parser.parse_args()

input_filename = f'./{args.input_filename}'
train_idx_filename = f'codeforces_A_gen_{args.model_fam}{args.idx_fileid}_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

time_ent_scores = []
space_ent_scores = []

time_all = []
space_all = []

time_all_list = []
space_all_list = []
for file_path, indices in zip(file_paths, train_idx):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    time_complexity_list = []
    space_complexity_list = []
    for i in indices:
        input_file = f'parsed_description_{args.model}_{fileid}_solution_{args.input_id}_{i}_{pid}.json'
        data = json.load(open(input_file))
        time_complexity = extract_complexity(data['time_complexity'])
        space_complexity = extract_complexity(data['space_complexity'])

        time_complexity_list.append(time_complexity)
        space_complexity_list.append(space_complexity)

    time_all += time_complexity_list
    space_all += space_complexity_list        
    
    time_all_list.append(time_complexity_list)
    space_all_list.append(space_complexity_list)
    
    time_entropy = entropy(time_complexity_list)
    space_entropy = entropy(space_complexity_list)
    time_ent_scores.append(time_entropy)
    space_ent_scores.append(space_entropy)
    print(f"entropy: {time_entropy}, {space_entropy} of {fileid} {pid}")

print(np.unique(time_all))
print(np.unique(space_all))
time_all_list = np.array(time_all_list)
space_all_list = np.array(space_all_list)
out_file = f'complexity_train_solution_entropy_gen_final_{args.model}_{args.input_id}.npz'
print(out_file)
np.savez(out_file, time_complexity_entropy=time_ent_scores, space_complexity_entropy=space_ent_scores,
         time_all=time_all_list, space_all=space_all_list)