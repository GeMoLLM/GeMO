set -x
python merge.py --prefix goodreads_completions_$1_llama-2-13b-chat_500_temp-$2-0 --n 5
python merge.py --prefix goodreads_completions_$1_llama-2-13b-chat_500_temp-$2-1 --n 5
python merge.py --prefix goodreads_completions_$1_llama-2-13b-chat_500_temp-$2-2 --n 5
python merge.py --prefix goodreads_completions_$1_llama-2-13b-chat_500_temp-$2-3 --n 5
python merge.py --prefix goodreads_completions_$1_llama-2-13b-chat_500_temp-$2-4 --n 5
python merge_goodreads_gen_reviews.py --input_prefix goodreads_completions_$1_llama-2-13b-chat_500_temp-$2 \
    --output_folder goodreads_$1_llama-2-13b-chat_temp-$2
python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_$1_llama-2-13b-chat_temp-$2 \
    --output_folder ../SafeNLP/goodreads_$1_adj_llama-2-13b-chat_temp-$2
python calculate_adj_sim_goodreads_gen.py --input_gen_folder ../SafeNLP/goodreads_$1_adj_llama-2-13b-chat_temp-$2 \
    --out_file pairwise_sim_scores_goodreads_$1_temp-$2.json
