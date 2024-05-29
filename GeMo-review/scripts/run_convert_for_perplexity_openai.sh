set -x
python convert_for_perplexity.py \
    --input_prefix goodreads_completions_$1_gpt-3.5-instruct-chat_500_temp-$2-p-$3 \
    --output_dir perplexity_goodreads_completions_$1_gpt-3.5-instruct-chat_500_temp-$2-p-$3