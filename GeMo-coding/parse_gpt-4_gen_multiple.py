import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Parse GPT-4 generated code')
parser.add_argument('--input_prefix', type=str, default='valid_solution')
args = parser.parse_args()

for fileid in tqdm(range(81)):

    filename = f'{args.input_prefix}_{fileid}.txt'

    lines = open(filename).readlines()

    cnt = 0
    for i in range(len(lines)):
        if '```python' in lines[i].strip():
            cnt += 1
            break

    print(cnt)