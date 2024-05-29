#!/bin/bash
set -x
for i in $(seq $3 $4);
do
  python generation_train_code_gpt4_final.py --output_id $1_$i --temperature $2 --input_filename '/scratch/fanw6/code_contests/codeforces_A_file_paths_sel_temp-0.5.txt'
done