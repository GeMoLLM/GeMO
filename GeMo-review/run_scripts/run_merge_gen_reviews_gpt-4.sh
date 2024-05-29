set -x
t=1.2
p=1.0
python merge_goodreads_gen_reviews.py \
    --input_prefix goodreads_completions_$1_$2-chat_500_temp-${t}-p-${p} \
    --output_folder folder_goodreads_completions_$1_$2-chat_500_temp-${t}-p-${p}
