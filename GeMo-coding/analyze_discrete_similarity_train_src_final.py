import argparse
import numpy as np
from itertools import combinations
import json
from utils import jaccard_similarity_index, regularize_algorithms, regularize_ds

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--field', type=str, default='algorithms')
args = parser.parse_args()

input_filename = './codeforces_A_file_paths_final.txt'
train_idx_filename = 'codeforces_A_train_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

sim_scores = []

all_data = []
for file_path, indices in zip(file_paths, train_idx):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    data_list = []
    for i in indices:
        input_file = f'parsed_alg_ds_{args.model}_{fileid}_{pid}_solution_{i}.json'
        cur_data = json.load(open(input_file))[args.field].split(', ')
        if args.field == 'algorithms':
            cur_data = regularize_algorithms(cur_data)
        elif args.field == 'data_structures':
            cur_data = regularize_ds(cur_data)
        all_data += cur_data
        data_list.append(set(list(np.unique(cur_data))))
        
    similarity = jaccard_similarity_index(data_list)
    sim_scores.append(similarity)
    print(f"similarity index: {similarity} of {fileid} {pid}")
    
print(np.unique(all_data))

out_file = f'{args.field}_train_solution_similarity_{args.model}_final.npy'
print(out_file)
np.save(out_file, np.array(sim_scores))