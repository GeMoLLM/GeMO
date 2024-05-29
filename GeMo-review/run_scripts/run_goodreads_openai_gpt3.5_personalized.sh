#!/bin/bash
set -x
for i in $(seq $3 $4);
do
    python generation_goodreads_openai.py \
        --temperature $1 \
        --top_p $2 \
        --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-$1-p-$2-$i;
done