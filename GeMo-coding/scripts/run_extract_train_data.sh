#!/bin/bash
set -x
for i in {00000..00127}
do
  bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-$i-of-00128 train-$i
done