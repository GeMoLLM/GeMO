#!/bin/bash
set -x
for i in $(seq $3 $4);
do
  python $1 --input_id $2_$i $5
done