import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Format check for descriptions')
parser.add_argument('--file_id', type=str, default='codeonly')
args = parser.parse_args()

indexs = np.load('./indexs.npy')
indices = np.where(indexs == 'A')[0]

for i in indices:
    for j in range(1, 6):
        filename = f'description_valid_solution_{args.file_id}_{j}_{i}.txt'
        lines = open(filename).readlines()
        filtered_lines = []
        for line in lines:
            if line.strip() != '':
                filtered_lines.append(line)
                
        assert len(filtered_lines) == 6, f'{i}, {j}, {filename}'
        
        assert filtered_lines[0].startswith('Description: '), f'{i}, {j}, {filename}'
        assert filtered_lines[1].startswith('Functionality: '), f'{i}, {j}, {filename}'
        assert filtered_lines[2].startswith('Algorithm: '), f'{i}, {j}, {filename}'
        assert filtered_lines[3].startswith('Data structure: '), f'{i}, {j}, {filename}'
        assert filtered_lines[4].startswith('Time complexity: '), f'{i}, {j}, {filename}'
        assert filtered_lines[5].startswith('Space complexity: '), f'{i}, {j}, {filename}'