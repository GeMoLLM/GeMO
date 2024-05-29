#!/bin/bash
set -x

# Define an array of celebrity names
celebrities=("Neil deGrasse Tyson" "Margaret Atwood" "David Attenborough" "Malala Yousafzai" "Jordan Peele")

# Loop through the array and generate commands
for i in "${!celebrities[@]}"; do
    person_name="${celebrities[$i]}"
    j=$((i+5))

    python generation_goodreads_openai.py \
        --temperature $1 \
        --top_p $2 \
        --person "$person_name" \
        --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-$1-p-$2-$j;
done
