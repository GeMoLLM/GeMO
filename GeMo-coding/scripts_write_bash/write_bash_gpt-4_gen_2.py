import json
import os

for i in range(81):  # Loop through the suffixes from 0 to 80
    json_file = f"valid_problem_{i}.json"
    bash_script = f"run_bazel_gpt4_gen_2_{i}.sh"

    with open(json_file, "r") as file:
        data = json.load(file)
        problem_name = data["name"]

    with open(bash_script, "w") as bash_file:
        bash_file.write("#!/bin/bash\n\n")

        bash_file.write(
            f'bazel run -c opt execution:solve_example_10 --   '
            f'--valid_path=/home/fanw6/main/dm-code_contests/code_contests_valid.riegeli '
            f'--problem_name="{problem_name}" '
            f'--solution_path="valid_solution_codeonly_{i}.py"\n'
        )

    # Make the bash script executable
    os.chmod(bash_script, 0o755)
