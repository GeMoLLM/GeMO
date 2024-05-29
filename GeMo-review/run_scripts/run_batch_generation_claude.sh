#!/bin/bash
set -x
for i in $(seq $4 $5);
do
  python generation_train_code_claude.py --output_id $1_$i --temperature $2 --top_p $3 $6
done