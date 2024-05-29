import json
import os

input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[100:135]

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    fileidno = fileid.split('-')[-1]
    pid = file_path.split('_')[-1].split('.')[0]
    
    json_file = f'{fileid}_problem_info_{pid}.json'

    bash_script = f"run_autojudge_{fileid}_{pid}.sh"

    with open(json_file, "r") as file:
        data = json.load(file)
        problem_name = data["name"].replace(' ', '_')
        
    with open(bash_script, "w") as bash_file:
        bash_file.write("#!/bin/bash\n\n")

        bash_file.write(
            f'bash run_judge.sh {fileid} {pid} "{problem_name}" $1'
        )

    # Make the bash script executable
    os.chmod(bash_script, 0o755)