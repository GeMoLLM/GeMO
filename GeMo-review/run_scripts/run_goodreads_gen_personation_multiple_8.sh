#!/bin/bash
set -x

# Define an array of celebrity names
celebrities=("Malala Yousafzai")

# Loop through the array and generate commands
for i in "${!celebrities[@]}"; do
    person_name="${celebrities[$i]}"

    python generation_goodreads.py \
        --temperature $3 \
        --person "$person_name" \
        --model_path $1 \
        --top_p $4 \
        --top_k $5 \
        --output_path goodreads_completions_personation_$2-chat_500_temp-$3_p-$4_k-$5-8;
done
