set -x
for t in 0.5 0.8 1.0 1.2; do
  for p in 0.90 0.95 0.98 1.00; do
    python merge_goodreads_gen_reviews.py \
        --input_prefix goodreads_completions_$1_$2-chat_500_temp-${t}-p-${p} \
        --output_folder folder_goodreads_completions_$1_$2-chat_500_temp-${t}-p-${p};\
  done
done