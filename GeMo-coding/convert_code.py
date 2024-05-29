import argparse
import json        

parser = argparse.ArgumentParser(description="Convert code to multi-line format.")
parser.add_argument("--input_file", type=str, default="valid_problem_0.json")
args = parser.parse_args()

def convert_code(solution_code):
    solution_code = solution_code.replace("\\\\", "PLACEHOLDER")
    # solution_code = solution_code.replace("\\\\", "\\")
    solution_code = solution_code.replace("\\n", "\n")
    solution_code = solution_code.replace("\\t", "\t")
    solution_code = solution_code.replace("\\r", "\r")
    solution_code = solution_code.replace("\\\'", "\'")
    solution_code = solution_code.replace("\\\"", "\"")
    solution_code = solution_code.replace("PLACEHOLDER", "\\")
    return solution_code

data = json.load(open(args.input_file))

out_prefix = args.input_file.split(".")[0]

for i in range(5):
    input_code = data[f'solutions-{i}-solution']
    print(input_code)
    output_code = convert_code(input_code)[1:-1]
    print(output_code)
    # output_code = "your_output_code_here"

    # Write the output code to a file
    with open(f'./{out_prefix}_correct_{i}.txt', 'w') as file:
        file.write(output_code)

for i in range(5):
    input_code = data[f'incorrect_solutions-{i}-solution']
    print(input_code)
    output_code = convert_code(input_code)[1:-1]
    print(output_code)
    # output_code = "your_output_code_here"

    # Write the output code to a file
    with open(f'./{out_prefix}_incorrect_{i}.txt', 'w') as file:
        file.write(output_code)
