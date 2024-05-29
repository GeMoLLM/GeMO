input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

N = 20

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    out_file = f'run_moss_traincode_src_{fileid}_{pid}.sh'

    with open(out_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -x\n')
        for i in range(N):
            for j in range(i+1, N):
                file1 = f'{fileid}_{pid}_solutions_{i}.txt'
                file2 = f'{fileid}_{pid}_solutions_{j}.txt'
                log_file = f'moss_results/moss_{fileid}_{pid}_solutions_{i}_{j}.log'
                f.write(f'time ./moss -l python {file1} {file2} > {log_file}\n')