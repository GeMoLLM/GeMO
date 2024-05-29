import re
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse description to json')
parser.add_argument('--model', type=str, default='gpt4')
parser.add_argument('--model_fam', type=str, default='gpt4')
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--idx_fileid', type=str, default='')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
args = parser.parse_args()

def parse_file_to_json(file_content):
    file_content = file_content.replace('\\n', '\n').replace('#', '')
    
    pattern = r'(Algorithms|Data Structures):\s*(.*)'
    parsed_data = {}

    for line in file_content.splitlines():
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            key = match.group(1).lower().replace(' ', '_')
            value = match.group(2).strip()
            parsed_data[key] = value
            if len(parsed_data) == 2:
                break

    # Fill in missing keys with empty strings
    keys = ['algorithms', 'data_structures']
    for key in keys:
        if key not in parsed_data:
            parsed_data[key] = ''

    return parsed_data

input_filename = f'./{args.input_filename}'
train_idx_filename = f'codeforces_A_gen_{args.model_fam}{args.idx_fileid}_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

for file_path, indices in zip(file_paths, train_idx):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    for i in indices:
        filename = f'alg_ds_{args.model}_{fileid}_{pid}_gpt-4_{args.input_id}_{i}.txt' \
            if 'claude' not in args.input_id \
                else f'alg_ds_{args.model}_{fileid}_{pid}_{args.input_id}_{i}.txt'

        with open(filename, 'r') as file:
            file_content = file.read().replace('\\n', '\n')
            json_data = parse_file_to_json(file_content)
            json.dump(json_data, open(f'parsed_alg_ds_{args.model}_{fileid}_solution_{args.input_id}_{i}_{pid}.json', 'w'), indent=4)
