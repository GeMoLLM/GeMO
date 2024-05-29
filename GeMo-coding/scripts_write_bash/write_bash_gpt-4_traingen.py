import json
import os

input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    fileidno = fileid.split('-')[-1]
    pid = file_path.split('_')[-1].split('.')[0]
    
    json_file = f'{fileid}_problem_info_{pid}.json'

    bash_script = f"run_bazel_{fileid}_{pid}.sh"

    with open(json_file, "r") as file:
        data = json.load(file)
        problem_name = data["name"]
        
    solution_path = f'{fileid}_solution_$1_{pid}.py'

    with open(bash_script, "w") as bash_file:
        bash_file.write("#!/bin/bash\n\n")

        bash_file.write(
            f'bazel run -c opt execution:solve_example_10 --   '
            f'--valid_path=/home/fanw6/main/dm-code_contests/code_contests_train.riegeli-{fileidno}-of-00128 '
            f'--problem_name="{problem_name}" '
            f'--solution_path={solution_path}\n'
        )

    # Make the bash script executable
    os.chmod(bash_script, 0o755)
