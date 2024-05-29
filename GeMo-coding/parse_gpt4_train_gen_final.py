import argparse
from tqdm import tqdm
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser(description='Parse GPT-4 generated code')
parser.add_argument('--input_id', type=str, default='valid_solution')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
args = parser.parse_args()

input_filename = f'./{args.input_filename}'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

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
        
    if start == len(lines) and end == -1:
        start, end = 0, len(lines)
    else:
        assert start < end, f'{file_path} {start} {end} {len(lines)}'

    code = ''.join(lines[start:end])

    out_file_path = osp.join(folder, f'{fileid}_solution_{args.input_id}_{pid}.py')

    output = open(out_file_path, 'w')
    output.write(code)