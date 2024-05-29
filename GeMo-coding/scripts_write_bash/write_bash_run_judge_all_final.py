input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths_final.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

out_file = f'run_autojudge_all_final.sh'

with open(out_file, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('set -x\n')
    for file_path in file_paths:
        fileid = file_path.split('_')[0]
        pid = file_path.split('_')[-1].split('.')[0]
        f.write(f'bash run_autojudge_{fileid}_{pid}.sh $1\n')