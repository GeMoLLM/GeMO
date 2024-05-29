input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())

out_file = 'run_convert_train_code_all.sh'
with open(out_file, 'w') as f:
    f.write('#!/bin/bash\n')
    for file_path in file_paths:
        f.write(f'python convert_train_code.py --input_file {file_path}\n')