#!/bin/bash
set -x
for i in $(seq $4 $5);
do
  python generation_train_code_gpt4_final.py --output_id $1_$i --temperature $2 --top_p $3
done