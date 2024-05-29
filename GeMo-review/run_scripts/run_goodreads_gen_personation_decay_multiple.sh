#!/bin/bash
set -x

# Define an array of celebrity names
celebrities=("Trevor Noah" "Janelle Monáe" "Yuval Noah Harari" "Serena Williams" "Reshma Saujani" "Neil deGrasse Tyson" "Margaret Atwood" "David Attenborough" "Malala Yousafzai" "Jordan Peele")

# Loop through the array and generate commands
for i in "${!celebrities[@]}"; do
    person_name="${celebrities[$i]}"

    python generation_goodreads_decay.py \
        --person "$person_name" \
        --model_path $1 \
        --top_p $4 \
        --top_k $5 \
        --decay $3 \
        --output_path goodreads_completions_personation_$2-chat_500_decay-$3_p-$4_k-$5-$i;
done
