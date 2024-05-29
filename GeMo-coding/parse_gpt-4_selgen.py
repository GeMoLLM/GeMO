import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Parse GPT-4 generated code')
parser.add_argument('--input_prefix', type=str, default='valid_solution')
args = parser.parse_args()

indexs = np.load('./indexs.npy')
indices = np.where(indexs == 'A')[0]

for fileid in tqdm(indices):

    filename = f'{args.input_prefix}_{fileid}.txt'

    lines = open(filename).readlines()

    for i in range(len(lines)):
        if '```python' in lines[i].strip():
            start = i + 1
            break

    for i in range(start, len(lines)):
        if '```' in lines[i].strip():
            end = i
            break
        
    print(start, end)
    code = ''.join(lines[start:end])
    print(code)

    output = open(f'{args.input_prefix}_{fileid}.py', 'w')
    output.write(code)