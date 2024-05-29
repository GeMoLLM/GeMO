#!/bin/bash
set -x
for i in $(seq $3 $4);
do
  python generation_train_code_gpt4_final.py --output_id $1_$i --temperature $2
done