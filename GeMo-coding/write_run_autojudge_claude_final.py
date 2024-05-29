lines_final = open('codeforces_A_file_paths_claude_final.txt').readlines()

with open('run_autojudge_all_claude_final.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('set -x\n')
    for line in lines_final:
        fileid = line.split('_')[0]
        pid = line.split('_')[-1].split('.')[0]
        cmd = f'bash run_autojudge_{fileid}_{pid}.sh $1\n'
        f.write(cmd)