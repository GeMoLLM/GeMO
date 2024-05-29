set -x
python convert_for_perplexity.py \
    --input_prefix goodreads_completions_$1_$2-chat_500_temp-$3_p-$4_k-$5 \
    --output_dir perplexity_goodreads_completions_$1_$2-chat_500_temp-$3_p-$4_k-$5