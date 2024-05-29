#!/bin/bash
python parse_gpt4_selgen.py --input_prefix valid_solution_$1
python write_bash_gpt4_selgen.py --input_prefix valid_solution_$1 \
  --outfile_id gpt4_gen_$1
bash run_bazel_all_gpt4_selgen.sh gpt4_gen_$1
python analyze_gpt4_selgen_acc.py --input_prefix valid_solution_$1 \
  --outfile_id gpt4_gen_$1
