input_filename = './codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    for i in range(20):
        filename = f'description_{fileid}_{pid}_solution_{i}.txt'

        lines = open(filename).readlines()
        filtered_lines = []
        for line in lines:
            if line.strip() != '':
                filtered_lines.append(line)
        
        if len(filtered_lines) != 7:
            print(f'len(filtered_lines) != 7, {i}, {filename}')
        
        elif not filtered_lines[0].lower().startswith('description: '):
            print(f'line 0 does not start with "Description: ", {i}, {filename}')
            
        elif not filtered_lines[1].lower().startswith('functionality: '):
            print(f'line 1 does not start with "Functionality: ", {i}, {filename}')
            
        elif not filtered_lines[2].lower().startswith('algorithm: '):
            print(f'line 2 does not start with "Algorithm: ", {i}, {filename}')
            
        elif not filtered_lines[3].lower().startswith('data structure: '):
            print(f'line 3 does not start with "Data structure: ", {i}, {filename}')
            
        elif not filtered_lines[4].lower().startswith('time complexity: '):
            print(f'line 4 does not start with "Time complexity: ", {i}, {filename}')
            
        elif not filtered_lines[5].lower().startswith('space complexity: '):
            print(f'line 5 does not start with "Space complexity: ", {i}, {filename}')
            
        elif not filtered_lines[6].lower().startswith('tags: '):
            print(f'line 6 does not start with "Tags: ", {i}, {filename}')