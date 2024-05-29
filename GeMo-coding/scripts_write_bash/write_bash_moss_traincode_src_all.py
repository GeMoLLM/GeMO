input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

out_file = 'run_moss_traincode_src_all.sh'
with open(out_file, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('set -x\n')
    for file_path in file_paths:
        fileid = file_path.split('_')[0]
        pid = file_path.split('_')[-1].split('.')[0]
        bashfile_name = f'run_moss_traincode_src_{fileid}_{pid}.sh'
        f.write(f'time bash {bashfile_name}\n')