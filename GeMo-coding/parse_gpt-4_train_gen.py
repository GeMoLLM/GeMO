import argparse
from tqdm import tqdm
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser(description='Parse GPT-4 generated code')
parser.add_argument('--input_id', type=str, default='valid_solution')
parser.add_argument('--probs_bid', type=int, default=100)
parser.add_argument('--probs_eid', type=int, default=135)
args = parser.parse_args()

input_filename = './codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[args.probs_bid:args.probs_eid]

folder = './'

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    file_path = osp.join(folder, f'{fileid}_solution_{args.input_id}_{pid}.txt')
        
    lines = open(file_path).readlines()

    start, end = len(lines), -1
    for i in range(len(lines)):
        if '```python' in lines[i].strip():
            start = i + 1
            break

    for i in range(start, len(lines)):
        if '```' in lines[i].strip():
            end = i
            break
        
    assert start < end, f'{file_path} {start} {end}'
    code = ''.join(lines[start:end])

    out_file_path = osp.join(folder, f'{fileid}_solution_{args.input_id}_{pid}.py')

    output = open(out_file_path, 'w')
    output.write(code)