#!/bin/bash
set -x
for i in $(seq $2 $3);
do
  bash $1 $i
done