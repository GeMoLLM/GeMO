#!/bin/bash
set -x
for i in $(seq $3 $4);
do
  bash $1 $2_$i
done