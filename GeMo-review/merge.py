import argparse
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n', type=int, default=-1)
parser.add_argument('--begin_id', type=int, default=0)

args = parser.parse_args()

files = [f'{args.prefix}_{i}.json' for i in range(args.begin_id, args.n)]

for file in files:
    assert file in os.listdir('./'), f'{file} not found in current directory'

print(f'Total {len(files)} files with prefix {args.prefix}')
print(files)

data = []
for file in files:
    with open(file, 'r') as f:
        data += json.load(f)["completions"]
        
print(f'Found {len(data)} completions')
output_path = f'{args.prefix}_merged.json'

with open(output_path, 'w') as f:
    json.dump(data, f)
print('output written to', output_path)

for file in files:
    os.remove(file)