#!/bin/bash
set -x
for i in {0..80}; do
    python convert_code.py --input_file valid_problem_$i.json
done
