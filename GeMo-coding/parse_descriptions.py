import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Format check for descriptions')
parser.add_argument('--file_id', type=str, default='codeonly')
args = parser.parse_args()

indexs = np.load('./indexs.npy')
indices = np.where(indexs == 'A')[0]

def extract(st, prefix):
    return st[len(prefix):]

for i in indices:
    for j in range(1, 6):
        filename = f'description_valid_solution_{args.file_id}_{j}_{i}.txt'
        lines = open(filename).readlines()
        filtered_lines = []
        for line in lines:
            if line.strip() != '':
                filtered_lines.append(line)
        assert filtered_lines[0].startswith('Description: '), f'{i}, {j}, {filename}'
        assert filtered_lines[1].startswith('Functionality: '), f'{i}, {j}, {filename}'
        assert filtered_lines[2].startswith('Algorithm: '), f'{i}, {j}, {filename}'
        assert filtered_lines[3].startswith('Data structure: '), f'{i}, {j}, {filename}'
        assert filtered_lines[4].startswith('Time complexity: '), f'{i}, {j}, {filename}'
        assert filtered_lines[5].startswith('Space complexity: '), f'{i}, {j}, {filename}'
        d = {}
        d['description'] = extract(filtered_lines[0], 'Description: ')
        d['functionality'] = extract(filtered_lines[1], 'Functionality: ')
        d['algorithm'] = extract(filtered_lines[2], 'Algorithm: ')
        d['data_structure'] = extract(filtered_lines[3], 'Data structure: ')
        d['time_complexity'] = extract(filtered_lines[4], 'Time complexity: ')
        d['space_complexity'] = extract(filtered_lines[5], 'Space complexity: ')
        
        json.dump(d, open(f'parsed_description_valid_solution_{args.file_id}_{j}_{i}.json', 'w'))
        