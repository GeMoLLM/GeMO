import argparse
from tqdm import tqdm
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser(description='Parse GPT-4 generated code')
parser.add_argument('--input_id', type=str, default='valid_solution')
args = parser.parse_args()

input_filename = './codeforces_A_file_paths_sel_temp-0.5.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 5, f'len(file_paths)={len(file_paths)}'

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