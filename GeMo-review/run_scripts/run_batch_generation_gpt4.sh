#!/bin/bash
set -x
for i in $(seq $3 $4);
do
  python generation_train_code_gpt4.py --output_id $1_$i --temperature $2 --begin_id $5 --end_id $6
done