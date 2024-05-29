#!/bin/bash
set -x

# Define an array of celebrity names
celebrities=("Trevor Noah" "Janelle Monáe" "Yuval Noah Harari" "Serena Williams" "Reshma Saujani" "Neil deGrasse Tyson" "Margaret Atwood" "David Attenborough" "Malala Yousafzai" "Jordan Peele")

# Loop through the array and generate commands
for i in "${!celebrities[@]}"; do
    person_name="${celebrities[$i]}"

    python generation_goodreads_openai_gpt4.py \
        --temperature $1 \
        --top_p $2 \
        --person "$person_name" \
        --output_path goodreads_completions_personation_gpt-4-chat_500_temp-$1-p-$2-$i;
done
