set -x
python generation_goodreads_openai.py \
    --temperature $1 \
    --top_p $2 \
    --person "Trevor Noah" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-$1-p-$2-0;\
python generation_goodreads_openai.py \
    --temperature $1 \
    --top_p $2 \
    --person "Janelle Monáe" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-$1-p-$2-1;\
python generation_goodreads_openai.py \
    --temperature $1 \
    --top_p $2 \
    --person "Yuval Noah Harari" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-$1-p-$2-2;\
python generation_goodreads_openai.py \
    --temperature $1 \
    --top_p $2 \
    --person "Serena Williams" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-$1-p-$2-3;\
python generation_goodreads_openai.py \
    --temperature $1 \
    --top_p $2 \
    --person "Reshma Saujani" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-$1-p-$2-4;\
