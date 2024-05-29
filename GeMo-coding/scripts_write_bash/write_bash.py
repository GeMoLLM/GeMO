import json
import os

for i in range(81):  # Loop through the suffixes from 0 to 80
    json_file = f"valid_problem_{i}.json"
    bash_script = f"run_bazel_{i}.sh"

    with open(json_file, "r") as file:
        data = json.load(file)
        problem_name = data["name"]

    with open(bash_script, "w") as bash_file:
        bash_file.write("#!/bin/bash\n\n")

        # Write bazel run commands for correct and incorrect solutions
        for j in range(5):  # Loop through correct_0 to correct_4 and incorrect_0 to incorrect_4
            for prefix in ["correct", "incorrect"]:
                bash_file.write(
                    f'bazel run -c opt execution:solve_example_10 --   '
                    f'--valid_path=/home/fanw6/main/dm-code_contests/code_contests_valid.riegeli '
                    f'--problem_name="{problem_name}" '
                    f'--solution_path="valid_problem_{i}_{prefix}_{j}.txt"\n'
                )
        bash_file.write("\n")  # Add a newline for readability

    # Make the bash script executable
    os.chmod(bash_script, 0o755)
