#!/bin/bash
set -x
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00001-of-00128 train-00001
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00002-of-00128 train-00002
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00005-of-00128 train-00005
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00006-of-00128 train-00006
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00007-of-00128 train-00007
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00009-of-00128 train-00009
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00010-of-00128 train-00010
bazel run -c opt :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00014-of-00128 train-00014