import json
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_prefix', type=str, default='valid_solution')
parser.add_argument('--outfile_id', type=str, default='valid_solution')
args = parser.parse_args()

indexs = np.load('/scratch/fanw6/code_contests/indexs.npy')
indices = np.where(indexs == 'A')[0]

for i in indices:  # Loop through the suffixes from 0 to 80
    json_file = f"valid_problem_{i}.json"
    bash_script = f"run_bazel_{args.outfile_id}_{i}.sh"

    with open(json_file, "r") as file:
        data = json.load(file)
        problem_name = data["name"]

    with open(bash_script, "w") as bash_file:
        bash_file.write("#!/bin/bash\n\n")

        bash_file.write(
            f'bazel run -c opt execution:solve_example_10 --   '
            f'--valid_path=/home/fanw6/main/dm-code_contests/code_contests_valid.riegeli '
            f'--problem_name="{problem_name}" '
            f'--solution_path="{args.input_prefix}_{i}.py"\n'
        )

    # Make the bash script executable
    os.chmod(bash_script, 0o755)
