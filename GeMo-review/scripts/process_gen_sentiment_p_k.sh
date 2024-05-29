set -x
python merge.py --prefix goodreads_completions_$1_$3-chat_500_temp-$2_p-$4_k-$5-0 --n 5
python merge.py --prefix goodreads_completions_$1_$3-chat_500_temp-$2_p-$4_k-$5-1 --n 5
python merge.py --prefix goodreads_completions_$1_$3-chat_500_temp-$2_p-$4_k-$5-2 --n 5
python merge.py --prefix goodreads_completions_$1_$3-chat_500_temp-$2_p-$4_k-$5-3 --n 5
python merge.py --prefix goodreads_completions_$1_$3-chat_500_temp-$2_p-$4_k-$5-4 --n 5
python merge_goodreads_gen_reviews.py --input_prefix goodreads_completions_$1_$3-chat_500_temp-$2_p-$4_k-$5 \
    --output_folder goodreads_$1_$3-chat_temp-$2_p-$4_k-$5
python sentiment_goodreads.py \
    --input_folder ../SafeNLP/goodreads_$1_$3-chat_temp-$2_p-$4_k-$5 \
    --output_file ../SafeNLP/sentiment_goodreads_$1_$3-chat_temp-$2_p-$4_k-$5.json \
    --typ gen