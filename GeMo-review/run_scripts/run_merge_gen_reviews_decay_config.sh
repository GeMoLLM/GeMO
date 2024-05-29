set -x
for d in linear exponential; do
  for p in 0.90 0.95 0.98 1.00; do
    python merge_goodreads_gen_reviews.py \
        --input_prefix goodreads_completions_$1_$2-chat_500_decay-${d}-$3-$4_p-${p}_k-50 \
        --output_folder folder_goodreads_completions_$1_$2-chat_500_decay-${d}-$3-$4_p-${p}_k-50;\
  done
done