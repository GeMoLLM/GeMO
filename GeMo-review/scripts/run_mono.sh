# 1587
# 2363
# total # samples = 2363

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_all_completions_llama-2-13b-chat_20 --max_new_tokens 20

CUDA_VISIBLE_DEVICES=3 python generation_bold_desc.py --type gender --model_path lmsys/vicuna-13b-v1.5 --output_path desc_1_all_completions_lmsys-vicuna-13b-v1.5_20 --max_new_tokens 20

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc.py --type gender --model_path meta-llama/Llama-2-7b-chat-hf --output_path desc_1_all_completions_llama-2-7b-chat_20 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc.py --type gender --model_path lmsys/vicuna-7b-v1.5 --output_path desc_1_all_completions_lmsys-vicuna-7b-v1.5_20 --max_new_tokens 20


CUDA_VISIBLE_DEVICES=3 python generation_bold_desc.py --type gender --model_path tiiuae/falcon-7b-instruct --output_path desc_1_all_completions_falcon-7b-instruct_20 --max_new_tokens 20

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc.py --type gender --model_path HuggingFaceH4/zephyr-7b-alpha --output_path desc_1_all_completions_zephyr-7b-alpha_20 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=2 python generation_bold_desc.py --type gender --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path desc_1_all_completions_mistral-7b-instruct_20 --max_new_tokens 20



python merge.py --prefix desc_1_all_completions_llama-2-13b-chat_20 --n 15
python merge.py --prefix desc_1_all_completions_lmsys-vicuna-13b-v1.5_20 --n 15
python merge.py --prefix desc_1_all_completions_llama-2-7b-chat_20 --n 15
python merge.py --prefix desc_1_all_completions_lmsys-vicuna-7b-v1.5_20 --n 15
python merge.py --prefix desc_1_all_completions_zephyr-7b-alpha_20 --n 15
python merge.py --prefix desc_1_all_completions_mistral-7b-instruct_20 --n 15

python parse_3adj.py --data_path desc_1_all_completions_llama-2-13b-chat_20_merged.json --out_path parsed_desc_1_all_completions_llama-2-13b-chat_20.jsonl
python parse_3adj.py --data_path desc_1_all_completions_lmsys-vicuna-13b-v1.5_20_merged.json --out_path parsed_desc_1_all_completions_lmsys-vicuna-13b-v1.5_20.jsonl
python parse_3adj.py --data_path desc_1_all_completions_llama-2-7b-chat_20_merged.json --out_path parsed_desc_1_all_completions_llama-2-7b-chat_20.jsonl
python parse_3adj.py --data_path desc_1_all_completions_lmsys-vicuna-7b-v1.5_20_merged.json --out_path parsed_desc_1_all_completions_lmsys-vicuna-7b-v1.5_20.jsonl
python parse_3adj.py --data_path desc_1_all_completions_zephyr-7b-alpha_20_merged.json --out_path parsed_desc_1_all_completions_zephyr-7b-alpha_20.jsonl
python parse_3adj.py --data_path desc_1_all_completions_mistral-7b-instruct_20_merged.json --out_path parsed_desc_1_all_completions_mistral-7b-instruct_20.jsonl



CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_all_completions_llama-2-13b-chat_20_given --max_new_tokens 20

CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-7b-chat-hf --output_path desc_1_all_completions_llama-2-7b-chat_20_given --max_new_tokens 20


# negative_celebrity_names.txt

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path negative_celeb_completions_llama-2-13b-chat_20_given --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-7b-chat-hf --output_path negative_celeb_completions_llama-2-7b-chat_20_given --max_new_tokens 20

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --type gender --model_path lmsys/vicuna-7b-v1.5 --output_path negative_celeb_completions_lmsys-vicuna-7b-v1.5_20_given --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --type gender --model_path lmsys/vicuna-13b-v1.5 --output_path negative_celeb_completions_lmsys-vicuna-13b-v1.5_20_given --max_new_tokens 20


python merge.py --prefix negative_celeb_completions_llama-2-13b-chat_20_given --n 1
python merge.py --prefix negative_celeb_completions_llama-2-7b-chat_20_given --n 1

python merge.py --prefix negative_celeb_completions_lmsys-vicuna-7b-v1.5_20_given --n 1
python merge.py --prefix negative_celeb_completions_lmsys-vicuna-13b-v1.5_20_given --n 1

python parse_3adj.py --data_path negative_celeb_completions_llama-2-7b-chat_20_given_merged.json --out_path parsed_negative_celeb_completions_llama-2-7b-chat_20_given_merged.jsonl
python parse_3adj.py --data_path negative_celeb_completions_llama-2-13b-chat_20_given_merged.json --out_path parsed_negative_celeb_completions_llama-2-13b-chat_20_given_merged.jsonl

python parse_3adj.py --data_path negative_celeb_completions_lmsys-vicuna-7b-v1.5_20_given_merged.json --out_path parsed_negative_celeb_completions_lmsys-vicuna-7b-v1.5_20_given_merged.jsonl
python parse_3adj.py --data_path negative_celeb_completions_lmsys-vicuna-13b-v1.5_20_given_merged.json --out_path parsed_negative_celeb_completions_lmsys-vicuna-13b-v1.5_20_given_merged.jsonl

python parse_3adj.py --data_path negative_celeb_completions_lmsys-vicuna-13b-v1.5_20_given_merged.json --out_path parsed_negative_celeb_completions_lmsys-vicuna-13b-v1.5_20_given_merged.jsonl

# artificial_human_names.txt

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path artificial_human_completions_llama-2-13b-chat_20_given --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-7b-chat-hf --output_path artificial_human_completions_llama-2-7b-chat_20_given --max_new_tokens 20

python merge.py --prefix artificial_human_completions_llama-2-13b-chat_20_given --n 1
python merge.py --prefix artificial_human_completions_llama-2-7b-chat_20_given --n 1

python parse_3adj.py --data_path artificial_human_completions_llama-2-7b-chat_20_given_merged.json --out_path parsed_artificial_human_completions_llama-2-7b-chat_20_given_merged.jsonl
python parse_3adj.py --data_path artificial_human_completions_llama-2-13b-chat_20_given_merged.json --out_path parsed_artificial_human_completions_llama-2-13b-chat_20_given_merged.jsonl

# artificial_american_names.txt

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path artificial_american_completions_llama-2-13b-chat_20_given --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-7b-chat-hf --output_path artificial_american_completions_llama-2-7b-chat_20_given --max_new_tokens 20

python merge.py --prefix artificial_american_completions_llama-2-13b-chat_20_given --n 1
python merge.py --prefix artificial_american_completions_llama-2-7b-chat_20_given --n 1

python parse_3adj.py --data_path artificial_american_completions_llama-2-7b-chat_20_given_merged.json --out_path parsed_artificial_american_completions_llama-2-7b-chat_20_given_merged.jsonl
python parse_3adj.py --data_path artificial_american_completions_llama-2-13b-chat_20_given_merged.json --out_path parsed_artificial_american_completions_llama-2-13b-chat_20_given_merged.jsonl

# artificial_asian_names.txt

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path artificial_asian_completions_llama-2-13b-chat_20_given --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --type gender --model_path meta-llama/Llama-2-7b-chat-hf --output_path artificial_asian_completions_llama-2-7b-chat_20_given --max_new_tokens 20

# controversial_celebrity_names.txt

CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_given.py --model_path meta-llama/Llama-2-13b-chat-hf --output_path controversial_celeb_completions_llama-2-13b-chat_20_given --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --model_path meta-llama/Llama-2-13b-chat-hf --output_path controversial_celeb_completions_llama-2-13b-chat_20_given-2 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=1 python generation_bold_desc_given.py --model_path meta-llama/Llama-2-13b-chat-hf --output_path controversial_celeb_completions_llama-2-13b-chat_20_given-3 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=0 python generation_bold_desc_given.py --model_path meta-llama/Llama-2-13b-chat-hf --output_path controversial_celeb_completions_llama-2-13b-chat_20_given-4 --max_new_tokens 20
CUDA_VISIBLE_DEVICES=3 python generation_bold_desc_given.py --model_path meta-llama/Llama-2-13b-chat-hf --output_path controversial_celeb_completions_llama-2-13b-chat_20_given-5 --max_new_tokens 20



python merge.py --prefix artificial_asian_completions_llama-2-13b-chat_20_given --n 1
python merge.py --prefix artificial_asian_completions_llama-2-7b-chat_20_given --n 1

python parse_3adj.py --data_path artificial_asian_completions_llama-2-7b-chat_20_given_merged.json --out_path parsed_artificial_asian_completions_llama-2-7b-chat_20_given_merged.jsonl
python parse_3adj.py --data_path artificial_asian_completions_llama-2-13b-chat_20_given_merged.json --out_path parsed_artificial_asian_completions_llama-2-13b-chat_20_given_merged.jsonl


# openai (gpt-3.5-turbo)

python openai_completions.py --output_path negative_celeb_completions_openai_turbo_3.5
python merge.py --prefix OpenAI_completions_debug --n 4
python parse_3adj.py --data_path OpenAI_completions_debug_merged.json --out_path parsed_negative_celeb_completions_openai_turbo_3.5_merged.jsonl

# openai (gpt-3.5-turbo)

python openai_completions.py --output_path negative_celeb_completions_openai_4 --model_name gpt-4
python merge.py --prefix negative_celeb_completions_openai_4 --n 4
python parse_3adj.py --data_path negative_celeb_completions_openai_4_merged.json --out_path parsed_negative_celeb_completions_openai_4_merged.jsonl



python vader_adj_gen.py --input_path parsed_desc_1_all_completions_llama-2-13b-chat_20.jsonl \
 --output_path vader_parsed_desc_1_all_completions_llama-2-13b-chat_20.json

python wsn_adj_articles.py
python wsn_adj_gen.py --input_path parsed_desc_1_all_completions_llama-2-13b-chat_20.jsonl \
 --output_path wsn_parsed_desc_1_all_completions_llama-2-13b-chat_20.json


CUDA_VISIBLE_DEVICES=3 python generation_bold_desc.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_all_completions_llama-2-13b-chat_20_gen-10_temp-0 --max_new_tokens 100 \
    --n_gen 10 --temperature 0.0


CUDA_VISIBLE_DEVICES=1 python generation_bold_desc.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.3 --max_new_tokens 20 \
    --n_gen 3 --temperature 0.3

CUDA_VISIBLE_DEVICES=0 python generation_bold_desc.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.8 --max_new_tokens 20 \
    --n_gen 3 --temperature 0.8


python merge.py --prefix desc_1_all_completions_llama-2-13b-chat_20_gen-10_temp-0 --n 15

python parse_3adj.py --n 10 --data_path desc_1_all_completions_llama-2-13b-chat_20_gen-10_temp-0_merged.json --out_path parsed_desc_1_all_completions_llama-2-13b-chat_20_gen-10_temp-0.jsonl

python merge.py --prefix desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.3 --n 15
python parse_3adj.py --data_path desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.3_merged.json --out_path parsed_desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.3.jsonl

python merge.py --prefix desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.8 --n 15
python parse_3adj.py --data_path desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.8_merged.json --out_path parsed_desc_1_all_completions_llama-2-13b-chat_20_gen-3_temp-0.8.jsonl


CUDA_VISIBLE_DEVICES=3 python generation_bold_desc.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_all_completions_llama-2-13b-chat_20_gen-10_temp-0.8 --max_new_tokens 100 \
    --n_gen 10 --temperature 0.8

# 0122
CUDA_VISIBLE_DEVICES=1 python generation_bold_desc_constrained.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_constrained_completions_llama-2-13b-chat_20 --max_new_tokens 20
python merge.py --prefix desc_1_constrained_completions_llama-2-13b-chat_20 --n 15
python parse_3adj.py --n 3 --data_path desc_1_constrained_completions_llama-2-13b-chat_20_merged.json --out_path parsed_desc_1_constrained_completions_llama-2-13b-chat_20.jsonl


CUDA_VISIBLE_DEVICES=2 python generation_bold_desc_constrained.py --type gender --model_path meta-llama/Llama-2-13b-chat-hf --output_path desc_1_constrained_completions_llama-2-13b-chat_100_gen-10_temp-0 --max_new_tokens 100 \
    --n_gen 10 --temperature 0.0
python merge.py --prefix desc_1_constrained_completions_llama-2-13b-chat_100_gen-10_temp-0 --n 15
python parse_3adj.py --n 10 --data_path desc_1_constrained_completions_llama-2-13b-chat_100_gen-10_temp-0_merged.json --out_path parsed_desc_1_constrained_completions_llama-2-13b-chat_100_gen-10_temp-0.jsonl


python scraper_early.py
python scraper_early.py --input_file celeb_links_American_actresses.json --output_folder wiki_early_content/American_actresses

python check_empty.py -> wiki_early_content_american_actors_non_empty_files.txt


# calculate cosine similarity

python calculate_adj_sim.py

python calculate_adj_sim.py --type actresses

-> pairwise_sim_scores_American_{args.type}.json

- merged and filtered -> pairwise_sim_scores.json

# 0130 - movie review datasets

python split_amazon_review.py

python organize_amazon_review.py

python count_amazon_review.py

-> amazon_review_counts.json, amazon_review_counts.npy

-> filter out the products with >= 500 reviews -> amazon_review_product_500.npy

# unique adjectives

python calculate_adj_sim.py --unique

python calculate_adj_sim.py --type actresses --unique

-> pairwise_sim_scores_American_{args.type}_unique.json

- merged and filtered -> pairwise_sim_scores.jso

# 0131 - scrape info for actors

python scraper_early_addition.py 

python scraper_early_addition.py --input_file celeb_links_American_actresses.json --output_folder wiki_info/American_actresses




- list of names with > 0 corpus adj -> filter_data_corpus1.npy


python wsn_adj_early_articles.py
python wsn_adj_early_articles.py --input_folder wiki_early_adjs/American_actresses --output_folder wiki_early_sent_wsn/American_actresses 

python wsn_adj_gen.py --input_path parsed_desc_1_constrained_completions_llama-2-13b-chat_100_gen-10_temp-0.jsonl \
 --output_path wsn_parsed_desc_1_constrained_completions_llama-2-13b-chat_100_gen-10_temp-0.json

f'Write a comprehensive review of the book titled {title}: '
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --output_path goodreads_completions_llama-2-13b-chat_500-0
python merge.py --prefix goodreads_completions_llama-2-13b-chat_500-0 --n 5
python merge.py --prefix goodreads_completions_llama-2-13b-chat_500-1 --n 5
python merge.py --prefix goodreads_completions_llama-2-13b-chat_500-2 --n 5
python merge.py --prefix goodreads_completions_llama-2-13b-chat_500-3 --n 5
python merge.py --prefix goodreads_completions_llama-2-13b-chat_500-4 --n 5

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --output_path goodreads_completions_llama-2-13b-chat_500-1

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --output_path goodreads_completions_llama-2-13b-chat_500-2

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --output_path goodreads_completions_llama-2-13b-chat_500-3

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --output_path goodreads_completions_llama-2-13b-chat_500-4



python extract_adj_goodreads_source.py
python extract_adj_goodreads_gen.py

python calculate_adj_sim_goodreads.py
-> pairwise_sim_scores_goodreads.json


# f'Write a personalized review of the book titled {title}: '
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --output_path goodreads_completions_personalized_llama-2-13b-chat_500-1

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --output_path goodreads_completions_personalized_llama-2-13b-chat_500-2

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --output_path goodreads_completions_personalized_llama-2-13b-chat_500-3

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --output_path goodreads_completions_personalized_llama-2-13b-chat_500-4;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --output_path goodreads_completions_personalized_llama-2-13b-chat_500-0


python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500-0 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500-1 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500-2 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500-3 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500-4 --n 5

python merge_goodreads_gen_reviews.py --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500 \
    --output_folder goodreads_personalized_llama-2-13b-chat

python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat \
    --output_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat
                                    
python calculate_adj_sim_goodreads_gen.py --input_gen_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat \
    --out_file pairwise_sim_scores_goodreads_personalized.json

#### change temperature

f'Write a personalized review of the book titled {title}: ', temperature=0.5, temperature=0.8
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-1;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-2;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-0;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-4;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-4;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-0;\


CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --output_path goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.5-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.8-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path meta-llama/Llama-2-7b-chat-hf --output_path goodreads_completions_personalized_llama-2-7b-chat_500_temp-0.1-4;\
## vicuna-7b
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.1-4;\


CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.5-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-7b-v1.5 --output_path goodreads_completions_personalized_vicuna-7b-chat_500_temp-0.8-4;\

## vicuna-13b


CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5-4;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.1-1;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.5-4;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.1-3;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.8-4;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.1-4;\

## HuggingFaceH4/zephyr-7b-alpha
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.1-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.5-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path HuggingFaceH4/zephyr-7b-alpha --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-0.8-4;\

## mistralai/Mistral-7B-Instruct-v0.1
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.1-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.5-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --model_path mistralai/Mistral-7B-Instruct-v0.1 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-0.8-4;\

## meta-llama/Llama-2-{13b,7b}-chat-hf
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.8_p-0.98_k-500-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-13b-chat_500_temp-0.5_p-0.98_k-500-4;\


CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path meta-llama/Llama-2-7b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_llama-7b-chat_500_temp-1.5_p-0.98_k-500-4;\

## meta-llama/Llama-2-13b-chat-hf, personation
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.1-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.5-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500-4;\

# [running] 
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-100-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-100-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-100-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-100-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-100-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-100-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-100-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-100-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-100-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-100-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-100-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-100-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-100-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 100 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-100-4;\

# [TODO:] 
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-50-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-50-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-50-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-50-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.95 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.95_k-50-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-50-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-50-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-50-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-50-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.98 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.98_k-50-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-50-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-50-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-50-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-50-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 0.90 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-0.90_k-50-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-50-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-50-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-50-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-50-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path meta-llama/Llama-2-13b-chat-hf --top_p 1.00 --top_k 50 --output_path goodreads_completions_personation_llama-13b-chat_500_temp-0.8_p-1.00_k-50-4;\


# [done]
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-4;\

# [running ] HuggingFaceH4/zephyr-7b-alpha, more sampling
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --model_path HuggingFaceH4/zephyr-7b-alpha --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_zephyr-7b-chat_500_temp-1.5_p-0.98_k-500-4;\

# [running ] mistralai/Mistral-7B-Instruct-v0.1, more sampling
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 1.5 --model_path mistralai/Mistral-7B-Instruct-v0.1 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personalized_mistral-7b-chat_500_temp-1.5_p-0.98_k-500-4;\


## vicuna-13b, personation
# [running, 1, -1]
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --person "Trevor Noah" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.1-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --person "Janelle Monáe" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --person "Yuval Noah Harari" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --person "Serena Williams" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --person "Reshma Saujani" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.1-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Trevor Noah" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.5-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Janelle Monáe" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Yuval Noah Harari" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Serena Williams" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --person "Reshma Saujani" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.5-4;\

# [done]
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Trevor Noah" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.8-0;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Janelle Monáe" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.8-1;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Yuval Noah Harari" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Serena Williams" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --person "Reshma Saujani" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-0.8-4;\

CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Trevor Noah" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5-0;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Janelle Monáe" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5-1;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Yuval Noah Harari" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5-2;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Serena Williams" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 1.5 --person "Reshma Saujani" --model_path lmsys/vicuna-13b-v1.5 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5-4;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --person "Trevor Noah" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-0;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --person "Janelle Monáe" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-1;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --person "Yuval Noah Harari" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --person "Serena Williams" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-3;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 1.5 --person "Reshma Saujani" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.95 --top_k 100 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100-4;\

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Trevor Noah" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-0;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Janelle Monáe" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Yuval Noah Harari" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-2;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Serena Williams" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-3;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 1.5 --person "Reshma Saujani" --model_path lmsys/vicuna-13b-v1.5 --top_p 0.98 --top_k 500 --output_path goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500-4;\


python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-0 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-1 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-2 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-3 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5-4 --n 5


python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-0 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-1 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-2 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-3 --n 5
python merge.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8-4 --n 5

python merge_goodreads_gen_reviews.py --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5 \
    --output_folder goodreads_personalized_llama-2-13b-chat_temp-0.5

python merge_goodreads_gen_reviews.py --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8 \
    --output_folder goodreads_personalized_llama-2-13b-chat_temp-0.8

python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.5 \
    --output_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.5

python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.8 \
    --output_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.8

python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-1.5 \
    --output_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-1.5

python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-13b-chat_temp-1.5_p-0.95_k-100 \
    --output_folder ../SafeNLP/goodreads_personalized_adj_llama-13b-chat_temp-1.5_p-0.95_k-100
    
python extract_adj_goodreads_gen.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-13b-chat_temp-1.5_p-0.98_k-500 \
    --output_folder ../SafeNLP/goodreads_personalized_adj_llama-13b-chat_temp-1.5_p-0.98_k-500

python calculate_adj_sim_goodreads_gen.py --input_gen_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.5 \
    --out_file pairwise_sim_scores_goodreads_personalized_temp-0.5.json

python calculate_adj_sim_goodreads_gen.py --input_gen_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.8 \
    --out_file pairwise_sim_scores_goodreads_personalized_temp-0.8.json

# f"Write a book review for the book titled {title}, from the viewpoint of different personas, such as 'aspiring writer', 'history enthusiast', 'teenage sci-fi fan', or 'career-focused parent', etc. Be creative!"

CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.1 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.1-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.5-1;\
CUDA_VISIBLE_DEVICES=0 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.8-1;\

CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.1 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.1-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.5-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.8-2;\
CUDA_VISIBLE_DEVICES=1 python generation_goodreads.py --temperature 0.1 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.1-0;\


CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.1 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.1-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.5-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.8-3;\
CUDA_VISIBLE_DEVICES=2 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.8-0;\

CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.1 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.1-4;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.5-4;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.8 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.8-4;\
CUDA_VISIBLE_DEVICES=3 python generation_goodreads.py --temperature 0.5 --output_path goodreads_completions_diffpersona_llama-2-13b-chat_500_temp-0.5-0;\


bash process_gen_output.sh diffpersona 0.1
bash process_gen_output.sh diffpersona 0.5
bash process_gen_output.sh diffpersona 0.8



python extract_adj_goodreads_source.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub/ \
    --output_folder ../review_data/goodreads/grouped_adjs_long_sub/

python extract_adj_goodreads_source.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en/ \
    --output_folder ../review_data/goodreads/grouped_adjs_long_sub_en/

python calculate_adj_sim_goodreads_src.py


# 0208 - extract embeddings

## sentence transformer embedding
python extract_embedding_goodreads.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub/ \
    --out_file ../review_data/goodreads/embeddings_src_long_sub.npz
python extract_embedding_goodreads.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat \
    --out_file ../SafeNLP/embeddings_gen_personalized_01.npz
python extract_embedding_goodreads.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.5 \
    --out_file ../SafeNLP/embeddings_gen_personalized_05.npz
python extract_embedding_goodreads.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.8 \
    --out_file ../SafeNLP/embeddings_gen_personalized_08.npz

## word2vec glove embedding

python extract_embedding_goodreads_word2vec.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub/ \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_word2vec.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub/ \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_google.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub/ \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_glovewiki50.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub/ \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_glovewiki50.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub_en/ \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_en_word2vec.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat \
    --out_file ../SafeNLP/embeddings_gen_word2vec_personalized_01.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.5 \
    --out_file ../SafeNLP/embeddings_gen_word2vec_personalized_05.npz

python extract_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.8 \
    --out_file ../SafeNLP/embeddings_gen_word2vec_personalized_08.npz

# clustering 

python cluster_adj_embeddings.py \
    --in_file ../review_data/goodreads/embeddings_src_long_sub_word2vec.npz \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_word2vec_nclusters.json

python cluster_adj_embeddings.py \
    --in_file ../review_data/goodreads/embeddings_src_long_sub_en_word2vec.npz \
    --out_file ../review_data/goodreads/embeddings_src_long_sub_en_word2vec_nclusters.json

python cluster_adj_embeddings.py \
    --in_file ../SafeNLP/embeddings_gen_word2vec_personalized_01.npz \
    --out_file ../SafeNLP/embeddings_gen_word2vec_personalized_01_nclusters.json

python cluster_adj_embeddings.py \
    --in_file ../SafeNLP/embeddings_gen_word2vec_personalized_05.npz \
    --out_file ../SafeNLP/embeddings_gen_word2vec_personalized_05_nclusters.json

python cluster_adj_embeddings.py \
    --in_file ../SafeNLP/embeddings_gen_word2vec_personalized_08.npz \
    --out_file ../SafeNLP/embeddings_gen_word2vec_personalized_08_nclusters.json

# 0208 - overall sentiment

## sentiment -- transformers.pipeline
python sentiment_goodreads.py \
    --input_folder ../review_data/goodreads/grouped_reviews_sub/ \
    --output_file ../review_data/goodreads/grouped_reviews_sub_sentiment.json

python sentiment_goodreads.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en/ \
    --output_file ../review_data/goodreads/grouped_reviews_long_sub_en_sentiment.json

## src
python sentiment_goodreads.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en/ \
    --output_file ../review_data/goodreads/grouped_reviews_long_sub_en_sentiment.json

## gen
python sentiment_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.5 \
    --output_file ../SafeNLP/sentiment_goodreads_personalized_llama-2-13b-chat_temp-0.5.json \
    --typ gen

CUDA_VISIBLE_DEVICES=1 python sentiment_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.8 \
    --output_file ../SafeNLP/sentiment_goodreads_personalized_llama-2-13b-chat_temp-0.8.json \
    --typ gen

python sentiment_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat \
    --output_file ../SafeNLP/sentiment_goodreads_personalized_llama-2-13b-chat_temp-0.1.json \
    --typ gen


# extract full text embeddings

python extract_embedding_goodreads_full.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en/ \
    --out_file ../review_data/goodreads/embedding_full_grouped_reviews_long_sub_en.npz

# TODO: run
python extract_embedding_goodreads_full.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat \
    --out_file ../SafeNLP/embedding_full_goodreads_personalized_llama-2-13b-chat_01.npz \
    --typ gen

python extract_embedding_goodreads_full.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.5 \
    --out_file ../SafeNLP/embedding_full_goodreads_personalized_llama-2-13b-chat_05.npz \
    --typ gen

python extract_embedding_goodreads_full.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.8 \
    --out_file ../SafeNLP/embedding_full_goodreads_personalized_llama-2-13b-chat_08.npz \
    --typ gen

# extract topics

python bertopic_goodreads.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en/ \
    --output_file ../review_data/goodreads/bertopic_grouped_reviews_long_sub_en.json \
    --typ src
python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_llama-2-13b-chat_01.json \
    --typ gen
python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.5 \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_llama-2-13b-chat_05.json \
    --typ gen
python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_llama-2-13b-chat_temp-0.8 \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_adj_llama-2-13b-chat_08.json \
    --typ gen

# cluster all
python cluster_adj_all_embeddings.py \
    --in_file ../review_data/goodreads/embeddings_src_long_sub_en_word2vec.npz \
    --out_file ../review_data/goodreads/cluster_all_embeddings_src_long_sub_en_word2vec.json

python cluster_adj_all_embeddings.py \
    --in_file ../SafeNLP/embeddings_gen_word2vec_personalized_01.npz \
    --out_file ../SafeNLP/cluster_all_embeddings_gen_word2vec_personalized_01.json

python cluster_adj_all_embeddings.py \
    --in_file ../SafeNLP/embeddings_gen_word2vec_personalized_05.npz \
    --out_file ../SafeNLP/cluster_all_embeddings_gen_word2vec_personalized_05.json

python cluster_adj_all_embeddings.py \
    --in_file ../SafeNLP/embeddings_gen_word2vec_personalized_08.npz \
    --out_file ../SafeNLP/cluster_all_embeddings_gen_word2vec_personalized_08.json

## after unique on adj from 5 reviews, performing standardization and dimensionality reduction

python cluster_adj_all_embeddings.py \
    --in_file ../review_data/goodreads/embeddings_all_src_long_sub_en_word2vec.npz \
    --out_file ../review_data/goodreads/cluster_all_reduced_embeddings_src_long_sub_en_word2vec.json

python cluster_adj_all_embeddings.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_01.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_gen_word2vec_personalized_01.json \
    --preproc reduce

python cluster_adj_all_embeddings.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_05.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_gen_word2vec_personalized_05.json \
    --preproc reduce

python cluster_adj_all_embeddings.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_08.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_gen_word2vec_personalized_08.json \
    --preproc reduce

# extract embeddings for all reviews together

python extract_all_embedding_goodreads_word2vec.py --input_folder ../review_data/goodreads/grouped_adjs_long_sub_en/ \
    --out_file ../review_data/goodreads/embeddings_all_src_long_sub_en_word2vec.npz

python extract_all_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat \
    --out_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_01.npz

python extract_all_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.5 \
    --out_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_05.npz

python extract_all_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-0.8 \
    --out_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_08.npz

python extract_all_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-2-13b-chat_temp-1.5 \
    --out_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_15.npz

python extract_all_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-13b-chat_temp-1.5_p-0.95_k-100 \
    --out_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_15_p-0.95_k-100.npz

python extract_all_embedding_goodreads_word2vec.py --input_folder ../SafeNLP/goodreads_personalized_adj_llama-13b-chat_temp-1.5_p-0.98_k-500 \
    --out_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_15_p-0.98_k-500.npz

# sentiment for more scenarios
bash process_gen_sentiment.sh personalized 1.5 llama-2-13b

bash process_gen_sentiment.sh personalized 0.1 llama-2-7b
bash process_gen_sentiment.sh personalized 0.5 llama-2-7b
bash process_gen_sentiment.sh personalized 0.8 llama-2-7b
bash process_gen_sentiment.sh personalized 1.5 llama-2-7b

bash process_gen_sentiment.sh personalized 0.1 vicuna-7b
bash process_gen_sentiment.sh personalized 0.5 vicuna-7b
bash process_gen_sentiment.sh personalized 0.8 vicuna-7b
bash process_gen_sentiment.sh personalized 1.5 vicuna-7b

bash process_gen_sentiment.sh personalized 0.1 vicuna-13b
bash process_gen_sentiment.sh personalized 0.5 vicuna-13b
bash process_gen_sentiment.sh personalized 0.8 vicuna-13b
bash process_gen_sentiment.sh personalized 1.5 vicuna-13b

bash process_gen_sentiment_p_k.sh personalized 1.5 llama-13b 0.98 500
bash process_gen_sentiment_p_k.sh personalized 1.5 llama-13b 0.95 100

bash process_gen_sentiment_p_k.sh personalized 1.5 vicuna-13b 0.98 500
bash process_gen_sentiment_p_k.sh personalized 1.5 vicuna-13b 0.95 100


bash process_gen_sentiment.sh personalized 0.1 zephyr-7b
bash process_gen_sentiment.sh personalized 0.5 zephyr-7b
bash process_gen_sentiment.sh personalized 0.8 zephyr-7b
bash process_gen_sentiment.sh personalized 1.5 zephyr-7b


bash process_gen_sentiment.sh personation 0.1 llama-13b
bash process_gen_sentiment.sh personation 0.5 llama-13b
bash process_gen_sentiment.sh personation 0.8 llama-13b
bash process_gen_sentiment.sh personation 1.5 llama-13b
bash process_gen_sentiment_p_k.sh personation 1.5 llama-13b 0.98 500
bash process_gen_sentiment_p_k.sh personation 1.5 llama-13b 0.95 100

bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 0.90 100
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 0.95 100
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 0.98 100
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 1.00 100

bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 0.90 50
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 0.95 50
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 0.98 50
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_p_k.sh personation 0.8 llama-13b 1.00 50

bash process_gen_sentiment.sh personation 0.1 vicuna-13b
bash process_gen_sentiment.sh personation 0.5 vicuna-13b
bash process_gen_sentiment.sh personation 0.8 vicuna-13b
bash process_gen_sentiment.sh personation 1.5 vicuna-13b
bash process_gen_sentiment_p_k.sh personation 1.5 vicuna-13b 0.98 500
bash process_gen_sentiment_p_k.sh personation 1.5 vicuna-13b 0.95 100

# 0215 - cluster all embeddings -- 2 stages

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../review_data/goodreads/embeddings_all_src_long_sub_en_word2vec.npz \
    --out_file ../review_data/goodreads/cluster_all_reduced_embeddings_2stage_src_long_sub_en_word2vec.json \
    --preproc reduce

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_01.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_2stage_gen_word2vec_personalized_01.json \
    --preproc reduce

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_05.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_2stage_gen_word2vec_personalized_05.json \
    --preproc reduce

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_08.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_2stage_gen_word2vec_personalized_08.json \
    --preproc reduce

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_15.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_2stage_gen_word2vec_personalized_15.json \
    --preproc reduce

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_15_p-0.95_k-100.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_2stage_gen_word2vec_personalized_15_p-0.95_k-100.json \
    --preproc reduce

python cluster_adj_all_embeddings_2stage.py \
    --in_file ../SafeNLP/embeddings_all_gen_word2vec_personalized_15_p-0.98_k-500.npz \
    --out_file ../SafeNLP/cluster_all_reduced_embeddings_2stage_gen_word2vec_personalized_15_p-0.98_k-500.json \
    --preproc reduce

########################################
####    0217 - openai gpt-3.5-turbo-instruct-0914
########################################

python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-0;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-1;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-2;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-3;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-4;\

## personalized [done]
# tab 2
bash run_goodreads_openai_gpt3.5_personalized.sh 0.8 1.0 5 9
bash run_goodreads_openai_gpt3.5_personalized.sh 0.8 0.98 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 0.8 0.95 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 0.8 0.90 0 9

# tab 4
bash run_goodreads_openai_gpt3.5_personalized.sh 0.5 1.0 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 0.5 0.98 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 0.5 0.95 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 0.5 0.90 0 9

# tab 6
bash run_goodreads_openai_gpt3.5_personalized.sh 1.0 1.0 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 1.0 0.98 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 1.0 0.95 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 1.0 0.90 0 9

# tab 3
bash run_goodreads_openai_gpt3.5_personalized.sh 1.2 1.0 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 1.2 0.98 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 1.2 0.95 0 9
bash run_goodreads_openai_gpt3.5_personalized.sh 1.2 0.90 0 9

## personation [done]

bash run_goodreads_openai_gpt3.5_personation_5-9.sh 0.8 1.0
bash run_goodreads_openai_gpt3.5_personation_10.sh 0.8 0.98
bash run_goodreads_openai_gpt3.5_personation_10.sh 0.8 0.95
bash run_goodreads_openai_gpt3.5_personation_10.sh 0.8 0.90

bash run_goodreads_openai_gpt3.5_personation_10.sh 0.5 1.0
bash run_goodreads_openai_gpt3.5_personation_10.sh 0.5 0.98
bash run_goodreads_openai_gpt3.5_personation_10.sh 0.5 0.95
bash run_goodreads_openai_gpt3.5_personation_10.sh 0.5 0.90

bash run_goodreads_openai_gpt3.5_personation_10.sh 1.0 1.0
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.0 0.98
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.0 0.95
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.0 0.90

bash run_goodreads_openai_gpt3.5_personation_10.sh 1.2 1.0
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.2 0.98
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.2 0.95
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.2 0.90

# [0331] GPT-3.5-instruct for high temperature generation

bash run_goodreads_openai_gpt3.5_personalized.sh 1.5 1.0 0 9;\
bash run_goodreads_openai_gpt3.5_personalized.sh 1.5 0.98 0 9;\
bash run_goodreads_openai_gpt3.5_personalized.sh 1.5 0.95 0 9;\
bash run_goodreads_openai_gpt3.5_personalized.sh 1.5 0.90 5 9;\

bash run_goodreads_openai_gpt3.5_personation_10.sh 1.5 1.0;\
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.5 0.98;\
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.5 0.95;\
bash run_goodreads_openai_gpt3.5_personation_10.sh 1.5 0.90;\

# TODO: merge, calculate perplexity, get the index etc.

python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-0;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-1;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-2;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-3;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-4;\


python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-0;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-1;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-2;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-3;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-4;\

python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-0;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-1;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-2;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-3;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --output_path goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-4;\

bash process_gen_sentiment_openai.sh personalized 0.8 gpt-3.5-instruct 1.0
bash process_gen_sentiment_openai.sh personalized 1.5 gpt-3.5-instruct 1.0
CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_openai.sh personalized 1.5 gpt-3.5-instruct 0.95
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_openai.sh personalized 1.5 gpt-3.5-instruct 0.98


# TODO: personation for the above two temperature

python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --person "Trevor Noah" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-0;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --person "Janelle Monáe" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-1;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --person "Yuval Noah Harari" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-2;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --person "Serena Williams" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-3;\
python generation_goodreads_openai.py \
    --temperature 0.8 \
    --top_p 1.0 \
    --person "Reshma Saujani" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0-4;\

python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --person "Trevor Noah" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-0;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --person "Janelle Monáe" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-1;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --person "Yuval Noah Harari" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-2;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --person "Serena Williams" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-3;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 1.0 \
    --person "Reshma Saujani" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0-4;\

python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --person "Trevor Noah" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-0;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --person "Janelle Monáe" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-1;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --person "Yuval Noah Harari" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-2;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --person "Serena Williams" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-3;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.95 \
    --person "Reshma Saujani" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95-4;\

python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --person "Trevor Noah" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-0;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --person "Janelle Monáe" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-1;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --person "Yuval Noah Harari" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-2;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --person "Serena Williams" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-3;\
python generation_goodreads_openai.py \
    --temperature 1.5 \
    --top_p 0.98 \
    --person "Reshma Saujani" \
    --output_path goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98-4;\

bash process_gen_sentiment_openai.sh personation 0.8 gpt-3.5-instruct 1.0
bash process_gen_sentiment_openai.sh personation 1.5 gpt-3.5-instruct 1.0
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_openai.sh personation 1.5 gpt-3.5-instruct 0.95
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_openai.sh personation 1.5 gpt-3.5-instruct 0.98


python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500 \
    --output_dir perplexity_goodreads_personalized_llama-2-13b-chat_500

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.5 \
    --output_dir perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.5

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-0.8 \
    --output_dir perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.8

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5 \
    --output_dir perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-1.5

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100 \
    --output_dir perplexity_goodreads_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500 \
    --output_dir perplexity_goodreads_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500


CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-2-13b-chat_500/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.1.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-1.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-1.5.npy

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.8/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.8.npy

CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.5.npy

CUDA_VISIBLE_DEVICES=0 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-13b-chat_500_temp-1.5_p-0.95_k-100/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-1.5_p-0.95_k-100.npy

# TODO: 
CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-13b-chat_500_temp-1.5_p-0.98_k-500/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-1.5_p-0.98_k-500.npy


python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_llama-13b-chat_500_temp-0.1 \
    --output_dir perplexity_goodreads_personation_llama-13b-chat_500_temp-0.1

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_llama-13b-chat_500_temp-0.5 \
    --output_dir perplexity_goodreads_personation_llama-13b-chat_500_temp-0.5

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_llama-13b-chat_500_temp-0.8 \
    --output_dir perplexity_goodreads_personation_llama-13b-chat_500_temp-0.8

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_llama-13b-chat_500_temp-1.5 \
    --output_dir perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100 \
    --output_dir perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500 \
    --output_dir perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_llama-13b-chat_500_temp-0.1/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_llama-13b-chat_500_temp-0.1.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_llama-13b-chat_500_temp-0.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_llama-13b-chat_500_temp-0.5.npy

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_llama-13b-chat_500_temp-0.8/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_llama-13b-chat_500_temp-0.8.npy

CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5.npy

CUDA_VISIBLE_DEVICES=0 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5_p-0.95_k-100.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_llama-13b-chat_500_temp-1.5_p-0.98_k-500.npy

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-0.1 \
    --output_dir perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.1

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-0.5 \
    --output_dir perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.5

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-0.8 \
    --output_dir perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.8

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5 \
    --output_dir perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100 \
    --output_dir perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100

python convert_for_perplexity.py --personation \
    --input_prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500 \
    --output_dir perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.1/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.1.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.5.npy

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.8/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-0.8.npy

CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5.npy

CUDA_VISIBLE_DEVICES=0 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personation_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500.npy


python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.1 \
    --output_dir perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.1

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.5 \
    --output_dir perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.5

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-0.8 \
    --output_dir perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.8

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5 \
    --output_dir perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100 \
    --output_dir perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500 \
    --output_dir perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.1/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.1.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.5.npy

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.8/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-0.8.npy

CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5.npy

CUDA_VISIBLE_DEVICES=0 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500.npy

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0 \
    --output_dir perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0 \
    --output_dir perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95 \
    --output_dir perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98 \
    --output_dir perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0 \
    --output_dir perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0 \
    --output_dir perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95 \
    --output_dir perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95

python convert_for_perplexity.py \
    --input_prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98 \
    --output_dir perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98


CUDA_VISIBLE_DEVICES=0 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0.npy;\
CUDA_VISIBLE_DEVICES=0 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0.npy

CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95.npy;\
CUDA_VISIBLE_DEVICES=1 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98.npy

CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-0.8-p-1.0.npy;\
CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.0.npy

CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95.npy;\
CUDA_VISIBLE_DEVICES=3 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98/ \
    --output_path ../../SafeNLP/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98.npy


bash run_goodreads_openai_gen_personalized.sh 1.2 0.90
bash run_goodreads_openai_gen_personalized.sh 1.2 0.95
bash run_goodreads_openai_gen_personalized.sh 1.2 0.98
bash run_goodreads_openai_gen_personalized.sh 1.2 1.00

bash run_goodreads_openai_gen_personalized_5.sh 1.2 0.90
bash run_goodreads_openai_gen_personalized_5.sh 1.2 0.95
bash run_goodreads_openai_gen_personalized_5.sh 1.2 0.98
bash run_goodreads_openai_gen_personalized_5.sh 1.2 1.00

CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.90 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.90 100

CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.95 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.95 100

CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.98 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.98 100

CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 1.00 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 1.00 100


CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.90 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.90 100

CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.95 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.95 100

CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.98 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.98 100

CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 1.00 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personalized.sh meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 1.00 100



CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_openai.sh personation 1.2 gpt-3.5-instruct 0.90
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_openai.sh personation 1.2 gpt-3.5-instruct 0.95
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_openai.sh personation 1.2 gpt-3.5-instruct 0.98
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_openai.sh personation 1.2 gpt-3.5-instruct 1.00

CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_openai.sh personalized 1.2 gpt-3.5-instruct 0.90
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_openai.sh personalized 1.2 gpt-3.5-instruct 0.95
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_openai.sh personalized 1.2 gpt-3.5-instruct 0.98
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_openai.sh personalized 1.2 gpt-3.5-instruct 1.00


bash run_convert_for_perplexity_openai.sh personalized 1.2 0.90
bash run_convert_for_perplexity_openai.sh personalized 1.2 0.95
bash run_convert_for_perplexity_openai.sh personalized 1.2 0.98
bash run_convert_for_perplexity_openai.sh personalized 1.2 1.00

bash run_convert_for_perplexity_openai.sh personation 1.2 0.90
bash run_convert_for_perplexity_openai.sh personation 1.2 0.95
bash run_convert_for_perplexity_openai.sh personation 1.2 0.98
bash run_convert_for_perplexity_openai.sh personation 1.2 1.00

CUDA_VISIBLE_DEVICES=0 bash run_calc_perplexity_openai.sh personalized 1.2 0.90;\
CUDA_VISIBLE_DEVICES=0 bash run_calc_perplexity_openai.sh personation 1.2 0.90

CUDA_VISIBLE_DEVICES=1 bash run_calc_perplexity_openai.sh personalized 1.2 0.95;\
CUDA_VISIBLE_DEVICES=1 bash run_calc_perplexity_openai.sh personation 1.2 0.95

CUDA_VISIBLE_DEVICES=2 bash run_calc_perplexity_openai.sh personalized 1.2 0.98;\
CUDA_VISIBLE_DEVICES=2 bash run_calc_perplexity_openai.sh personation 1.2 0.98

CUDA_VISIBLE_DEVICES=3 bash run_calc_perplexity_openai.sh personalized 1.2 1.00;\
CUDA_VISIBLE_DEVICES=3 bash run_calc_perplexity_openai.sh personation 1.2 1.00


CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 0.90 50
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 0.95 50
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 0.98 50
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 1.00 50

CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 0.90 100
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 0.95 100
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 0.98 100
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_p_k.sh personation 1.2 llama-2-13b 1.00 100

CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 0.95 50
CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 1.00 50
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 0.95 100
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 1.00 100

CUDA_VISIBLE_DEVICES=0 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 0.90 50
CUDA_VISIBLE_DEVICES=2 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 0.90 100

CUDA_VISIBLE_DEVICES=1 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 0.98 50
CUDA_VISIBLE_DEVICES=3 bash process_gen_sentiment_p_k.sh personalized 1.2 llama-2-13b 0.98 100


# topics for goodreads openai
python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.5-p-1.0/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.5_p-1.0.json \
    --typ gen
python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.5-p-0.98/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.5_p-0.98.json \
    --typ gen
python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.5-p-0.95/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.5_p-0.95.json \
    --typ gen

CUDA_VISIBLE_DEVICES=0 python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2-p-1.00/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2_p-1.00.json \
    --typ gen
CUDA_VISIBLE_DEVICES=1 python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2-p-0.98/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2_p-0.98.json \
    --typ gen
CUDA_VISIBLE_DEVICES=2 python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2-p-0.95/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2_p-0.95.json \
    --typ gen
CUDA_VISIBLE_DEVICES=3 python bertopic_goodreads.py \
    --input_folder ../SafeNLP/goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2-p-0.90/ \
    --output_file ../SafeNLP/bertopic_full_goodreads_personalized_gpt-3.5-instruct-chat_temp-1.2_p-0.90.json \
    --typ gen


CUDA_VISIBLE_DEVICES=2 python perplexity_calc.py \
    --data_path ../../SafeNLP/data_perplexity/perplexity_goodreads_personalized_llama-2-13b-chat_500/ \
    --output_path ../../SafeNLP/perplexity_goodreads_personalized_llama-2-13b-chat_500_temp-0.1.npy


bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 0.90 50
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 0.95 50
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 0.98 50
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 1.00 50
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 0.90 100
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 0.95 100
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 0.98 100
bash run_convert_for_perplexity.sh personalized llama-2-13b 1.2 1.00 100

bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 0.90 50
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 0.95 50
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 0.98 50
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 1.00 50
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 0.90 100
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 0.95 100
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 0.98 100
bash run_convert_for_perplexity.sh personation llama-2-13b 1.2 1.00 100



CUDA_VISIBLE_DEVICES=0 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 0.90 50;\
CUDA_VISIBLE_DEVICES=0 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 0.95 50;\
CUDA_VISIBLE_DEVICES=0 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 0.98 50;\
CUDA_VISIBLE_DEVICES=0 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 1.00 50;\

CUDA_VISIBLE_DEVICES=1 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 0.90 100;\
CUDA_VISIBLE_DEVICES=1 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 0.95 100;\
CUDA_VISIBLE_DEVICES=1 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 0.98 100;\
CUDA_VISIBLE_DEVICES=1 bash run_calc_perplexity.sh personalized llama-2-13b 1.2 1.00 100;\

CUDA_VISIBLE_DEVICES=2 bash run_calc_perplexity.sh personation llama-2-13b 1.2 0.90 50;\
CUDA_VISIBLE_DEVICES=2 bash run_calc_perplexity.sh personation llama-2-13b 1.2 0.95 50;\
CUDA_VISIBLE_DEVICES=2 bash run_calc_perplexity.sh personation llama-2-13b 1.2 0.98 50;\
CUDA_VISIBLE_DEVICES=2 bash run_calc_perplexity.sh personation llama-2-13b 1.2 1.00 50;\

CUDA_VISIBLE_DEVICES=3 bash run_calc_perplexity.sh personation llama-2-13b 1.2 0.90 100;\
CUDA_VISIBLE_DEVICES=3 bash run_calc_perplexity.sh personation llama-2-13b 1.2 0.95 100;\
CUDA_VISIBLE_DEVICES=3 bash run_calc_perplexity.sh personation llama-2-13b 1.2 0.98 100;\
CUDA_VISIBLE_DEVICES=3 bash run_calc_perplexity.sh personation llama-2-13b 1.2 1.00 100;\


------------------------------------------------------------------------------------------------------------------------

bash run_goodreads_openai_gpt4_personalized.sh 1.0 0.90 0 9

bash run_goodreads_openai_gpt4_personation_10.sh 1.0 0.90


export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 1.00 50 5 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 1.00 50 5 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 1.00 50 5 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 1.00 50 5 9;\

----------------------------------------------------------------------------------------------------------------
# [done]
export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 1.00 50 5 9;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 1.00 50 5 9;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 1.00 50 5 9;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.90 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.95 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.98 50 5 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 1.00 50 5 9;\

----------------------------------------------------------------------------------------------------------------
# progress
# done: llama personalized 5-9 (merged)
# done: vicuna personalized 5-9 (merged)
# done: llama personalized 0-4 (merged)
# done: vicuna personalized 0-4 (merged)

# done: llama personation 5-9 (merged)
# done: vicuna personation 5-9 (merged)
# done: llama personation 0-4 (merged)
# done: vicuna personation 0-4 (merged)

# TODO: other models?


----------------------------------------------------------------------------------------------------------------
# previous missing configs to run

# [done]

# llama-2-13b personalized 0-4
export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 1.00 50 0 4;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 1.00 50 0 4;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 1.00 50 0 4;\


----------------------------------------------------------------------------------------------------------------
# llama-2-13b personation 5-9
# [done]
export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.90 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.95 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.98 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.90 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.95 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.98 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.90 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.95 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.98 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.90 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.95 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 0.98 50;\
bash run_goodreads_gen_personation_multiple_5-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.2 1.00 50;\

----------------------------------------------------------------------------------------------------------------
python change_name_format.py 



----------------------------------------------------------------------------------------------------------------
# [done]
# vicuna-13b personalized 0-4
export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 1.00 50 0 4;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 1.00 50 0 4;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 1.00 50 0 4;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.90 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.95 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.98 50 0 4;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 1.00 50 0 4;\

----------------------------------------------------------------------------------------------------------------
# merge generated files
bash run_merge_goodreads_overall.sh personalized llama-2-13b 
bash run_merge_goodreads_overall.sh personalized vicuna-13b 5 9
bash run_merge_goodreads_overall.sh personalized vicuna-13b 0 4
bash run_merge_goodreads_overall.sh personalized llama-2-13b 0 4

bash run_merge_goodreads_overall.sh personation llama-2-13b 5 9
bash run_merge_goodreads_overall.sh personation vicuna-13b 5 9
bash run_merge_goodreads_overall.sh personation llama-2-13b 0 4
bash run_merge_goodreads_overall.sh personation vicuna-13b 0 4

# [0324] merge decay files
bash run_merge_goodreads_decay_overall.sh personalized llama-2-13b 0 9
bash run_merge_goodreads_decay_overall.sh personalized vicuna-13b 0 9
bash run_merge_goodreads_decay_overall.sh personation llama-2-13b 0 9
bash run_merge_goodreads_decay_overall.sh personation vicuna-13b 0 9

# [0401] merge decay files config (1.2, 50)
bash run_merge_goodreads_decay_config_overall.sh personalized llama-2-13b 1.2 50 0 9
bash run_merge_goodreads_decay_config_overall.sh personalized vicuna-13b 1.2 50 0 9
bash run_merge_goodreads_decay_config_overall.sh personation llama-2-13b 1.2 50 0 9
bash run_merge_goodreads_decay_config_overall.sh personation vicuna-13b 1.2 50 0 9

# [0324] merge gpt-3.5-instruct files
bash run_merge_goodreads_gpt-3.5-instruct_overall.sh personalized 0 9
bash run_merge_goodreads_gpt-3.5-instruct_overall.sh personation 0 9

# [0329] merge gpt-4 personalized files
bash run_merge_goodreads_gpt-4_overall.sh personalized 0 9
# [0331] merged gpt-4 personation files
bash run_merge_goodreads_gpt-4_overall.sh personation 0 9

# [0331] merge llama and vicuna files for temp=1.5
bash run_merge_goodreads_temp-1.5.sh personation llama-2-13b 0 9
bash run_merge_goodreads_temp-1.5.sh personation vicuna-13b 0 9
bash run_merge_goodreads_temp-1.5.sh personalized llama-2-13b 0 9
bash run_merge_goodreads_temp-1.5.sh personalized vicuna-13b 0 9

# check missing files
python check_missing_files.py --mode personalized --model llama-2-13b
python check_missing_files.py --mode personalized --model vicuna-13b

# [0405] merge nonchat files
bash run_merge_nonchat_goodreads_overall.sh personalized llama-2-13b 0 9
bash run_merge_nonchat_goodreads_overall.sh personation llama-2-13b 0 9

----------------------------------------------------------------------------------------------------------------
# merge generated files

bash run_merge_gen_reviews.sh personalized llama-2-13b
bash run_merge_gen_reviews.sh personalized vicuna-13b
bash run_merge_gen_reviews.sh personation llama-2-13b
bash run_merge_gen_reviews.sh personation vicuna-13b

# [0325] merge generated files for decay
bash run_merge_gen_reviews_decay.sh personalized llama-2-13b;\
bash run_merge_gen_reviews_decay.sh personalized vicuna-13b;\
bash run_merge_gen_reviews_decay.sh personation llama-2-13b;\
bash run_merge_gen_reviews_decay.sh personation vicuna-13b;\

# [0401] merge generated files for decay (1.2, 50)
bash run_merge_gen_reviews_decay_config.sh personalized llama-2-13b 1.2 50;\
bash run_merge_gen_reviews_decay_config.sh personalized vicuna-13b 1.2 50;\
bash run_merge_gen_reviews_decay_config.sh personation llama-2-13b 1.2 50;\
bash run_merge_gen_reviews_decay_config.sh personation vicuna-13b 1.2 50;\

# [0325] merge generated files for gpt-3.5-instruct

bash run_merge_gen_reviews_gpt-3.5-instruct.sh personalized gpt-3.5-instruct;\
bash run_merge_gen_reviews_gpt-3.5-instruct.sh personation gpt-3.5-instruct;\

# [0329] merge generated files for gpt-4
bash run_merge_gen_reviews_gpt-4.sh personalized gpt-4;\
# [0331] merge generated files for gpt-4
bash run_merge_gen_reviews_gpt-4.sh personation gpt-4;\

# [0331] merge llama and vicuna files for temp=1.5
bash run_merge_gen_reviews_temp-1.5.sh personalized llama-2-13b
bash run_merge_gen_reviews_temp-1.5.sh personalized vicuna-13b
bash run_merge_gen_reviews_temp-1.5.sh personation llama-2-13b
bash run_merge_gen_reviews_temp-1.5.sh personation vicuna-13b

# [0405] merge nonchat files
bash run_merge_gen_nonchat_reviews.sh personalized llama-2-13b
bash run_merge_gen_nonchat_reviews.sh personation llama-2-13b

----------------------------------------------------------------------------------------------------------------
# calculate perplexity via meta-llama/Llama-2-7b-hf

python calculate_perplexity.py --T_list 1.0,0.8,0.5,1.2 --P_list 0.90,0.95,0.98,1.00

python calculate_perplexity.py --indices 0,1,2,3,4,6,7,8,9

python calculate_perplexity.py --bid 0 --eid 10 --model vicuna-13b

python calculate_perplexity.py --bid 0 --eid 10 --model vicuna-13b --mode personation
python calculate_perplexity.py --bid 0 --eid 10 --model llama-2-13b --mode personation

python calculate_perplexity.py --T_list 1.5 --bid 0 --eid 10 -d 0
python calculate_perplexity.py --T_list 1.5 --mode personation --bid 0 --eid 10 -d 0
python calculate_perplexity.py --T_list 1.5 --model vicuna-13b --mode personation --bid 0 --eid 10 -d 3
python calculate_perplexity.py --T_list 1.5 --model vicuna-13b --mode personalized --bid 0 --eid 10 -d 3

# [0405] calculate perplexity for the nonchat
python calculate_perplexity_nonchat.py --model llama-2-13b --mode personation --bid 0 --eid 10 -d 3
python calculate_perplexity_nonchat.py --model llama-2-13b --mode personalized --bid 0 --eid 10 -d 1

# [0401] calculate perplexity for the source
python calculate_perplexity_src.py --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en_10/ -d 1
-> perplexity_goodreads_src.json

# [0324] calculate perplexity for decay
python calculate_perplexity_decay.py --bid 0 --eid 10 --model llama-2-13b --mode personalized -d 0
python calculate_perplexity_decay.py --bid 0 --eid 10 --model llama-2-13b --mode personation -d 1
python calculate_perplexity_decay.py --bid 0 --eid 10 --model vicuna-13b --mode personalized -d 2
python calculate_perplexity_decay.py --bid 0 --eid 10 --model vicuna-13b --mode personation -d 3

# [0401] calculate perplexity for decay (1.2, 50)
python calculate_perplexity_decay.py --bid 0 --eid 10 --model llama-2-13b --mode personalized -d 0 -et 1.2 -p 50
python calculate_perplexity_decay.py --bid 0 --eid 10 --model llama-2-13b --mode personation -d 1 -et 1.2 -p 50
python calculate_perplexity_decay.py --bid 0 --eid 10 --model vicuna-13b --mode personalized -d 2 -et 1.2 -p 50
python calculate_perplexity_decay.py --bid 0 --eid 10 --model vicuna-13b --mode personation -d 3 -et 1.2 -p 50

# [0324] calculate perplexity for gpt-3.5-instruct
python calculate_perplexity.py --bid 0 --eid 10 --model gpt-3.5-instruct --mode personation -d 0
python calculate_perplexity.py --bid 0 --eid 10 --model gpt-3.5-instruct --mode personalized -d 1

# [0401] calculate perplexity for gpt-4
python calculate_perplexity_gpt4.py --bid 0 --eid 10 --model gpt-4 --mode personation -d 2
python calculate_perplexity_gpt4.py --bid 0 --eid 10 --model gpt-4 --mode personalized -d 1

# [0330] examine perplexity scores for temp=1.5 (too high)
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5 --n 5 -d 0;\
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5 --n 5 -d 0;\
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5_p-0.95_k-100 --n 5 -d 0;\

python calculate_perplexity_high.py --prefix goodreads_completions_personation_llama-2-13b-chat_500_temp-1.5 --n 5 -d 1;\
python calculate_perplexity_high.py --prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5 --n 5 -d 1;\
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5_p-0.98_k-500 --n 5 -d 1;\
python calculate_perplexity_high.py --prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.00 --n 5 -d 1;\

python calculate_perplexity_high.py --prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.00 --n 5 -d 2;\
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98 --n 5 -d 2;\
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95 --n 5 -d 2;\
python calculate_perplexity_high.py --prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98 --n 5 -d 2;\

python calculate_perplexity_high.py --prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100 --n 5 -d 3;\
python calculate_perplexity_high.py --prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500 --n 5 -d 3;\
python calculate_perplexity_high.py --prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95 --n 5 -d 3;\


----------------------------------------------------------------------------------------------------------------
# combine perplexity score [0331 updated]

python examine_perplexity_scores.py 
-> perplexity_scores/perplexity_goodreads_completions_personalized_llama-2-13b-chat_500.npy

python examine_perplexity_scores.py --model vicuna-13b 
-> perplexity_scores/perplexity_goodreads_completions_personalized_vicuna-13b-chat_500.npy

python examine_perplexity_scores.py --mode personation
-> perplexity_scores/perplexity_goodreads_completions_personation_llama-2-13b-chat_500.npy

python examine_perplexity_scores.py --model vicuna-13b --mode personation
-> perplexity_scores/perplexity_goodreads_completions_personation_vicuna-13b-chat_500.npy

python examine_perplexity_scores.py --chat nonchat --T_list 1.2
python examine_perplexity_scores.py --chat nonchat --mode personation --T_list 1.2

# [0325] perplexity for the decay generations
python examine_perplexity_scores_decay.py --model llama-2-13b --mode personalized
python examine_perplexity_scores_decay.py --model llama-2-13b --mode personation;\
python examine_perplexity_scores_decay.py --model vicuna-13b --mode personalized;\
python examine_perplexity_scores_decay.py --model vicuna-13b --mode personation

# [0402] examine perplexity for decay (1.2, 50)
python examine_perplexity_scores_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50 
python examine_perplexity_scores_decay.py --model llama-2-13b --mode personation -et 1.2 -p 50
python examine_perplexity_scores_decay.py --model vicuna-13b --mode personalized -et 1.2 -p 50
python examine_perplexity_scores_decay.py --model vicuna-13b --mode personation -et 1.2 -p 50

# [0325] perplexity for the gpt-3.5-instruct generations
python examine_perplexity_scores_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation
-> perplexity_scores/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500.npy

python examine_perplexity_scores_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized
-> perplexity_scores/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500.npy

# [0401] perplexity for GPT-4
python examine_perplexity_scores_gpt-3.5-instruct.py --model gpt-4 --mode personation --T_list 1.2 --P_list 1.0
-> perplexity_scores/perplexity_goodreads_completions_personation_gpt-4-chat_500.npy
python examine_perplexity_scores_gpt-3.5-instruct.py --model gpt-4 --mode personalized --T_list 1.2 --P_list 1.0
-> perplexity_scores/perplexity_goodreads_completions_personalized_gpt-4-chat_500.npy

# [0330] perplexity for temp=1.5 (too high)
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5_p-0.95_k-100 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personation_llama-2-13b-chat_500_temp-1.5 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5_p-0.98_k-500 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.00 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.00 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500 --n 5;\
python examine_perplexity_scores_high.py --prefix goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95 --n 5;\

->
perplexity_scores/perplexity_goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5.npy
perplexity_scores/perplexity_goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5.npy
perplexity_scores/perplexity_goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5_p-0.95_k-100.npy
perplexity_scores/perplexity_goodreads_completions_personation_llama-2-13b-chat_500_temp-1.5.npy
perplexity_scores/perplexity_goodreads_completions_personation_vicuna-13b-chat_500_temp-1.5.npy
perplexity_scores/perplexity_goodreads_completions_personalized_llama-2-13b-chat_500_temp-1.5_p-0.98_k-500.npy
perplexity_scores/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-1.00.npy
perplexity_scores/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-1.00.npy
perplexity_scores/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98.npy
perplexity_scores/perplexity_goodreads_completions_personalized_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95.npy
perplexity_scores/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.98.npy
perplexity_scores/perplexity_goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.95_k-100.npy
perplexity_scores/perplexity_goodreads_completions_personalized_vicuna-13b-chat_500_temp-1.5_p-0.98_k-500.npy
perplexity_scores/perplexity_goodreads_completions_personation_gpt-3.5-instruct-chat_500_temp-1.5-p-0.95.npy

----------------------------------------------------------------------------------------------------------------
# extract indices given perplexity scores [0331 updated]

python obtain_indices.py
-> indices_dict_goodreads_personalized_llama-2-13b-chat_500.pkl

python obtain_indices.py --model vicuna-13b
-> indices_dict_goodreads_personalized_vicuna-13b-chat_500.pkl

python obtain_indices.py --mode personation
-> indices_dict_goodreads_personation_llama-2-13b-chat_500.pkl

python obtain_indices.py --model vicuna-13b --mode personation
-> indices_dict_goodreads_personation_vicuna-13b-chat_500.pkl

python obtain_indices.py --chat nonchat --model llama-2-13b --mode personation --T_list 1.2
python obtain_indices.py --chat nonchat --model llama-2-13b --mode personalized --T_list 1.2


# [0325] obtain indices for decay
python obtain_indices_decay.py;\
python obtain_indices_decay.py --model vicuna-13b;\
python obtain_indices_decay.py --mode personation;\
python obtain_indices_decay.py --model vicuna-13b --mode personation

# [0402] obtain indices for decay (1.2, 50)
python obtain_indices_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50
python obtain_indices_decay.py --model llama-2-13b --mode personation -et 1.2 -p 50
python obtain_indices_decay.py --model vicuna-13b --mode personalized -et 1.2 -p 50
python obtain_indices_decay.py --model vicuna-13b --mode personation -et 1.2 -p 50

# [0325] obtain indices for gpt-3.5-instruct
python obtain_indices.py --model gpt-3.5-instruct --mode personation
-> indices_dict_goodreads_personation_gpt-3.5-instruct-chat_500.pkl

python obtain_indices.py --model gpt-3.5-instruct --mode personalized
-> indices_dict_goodreads_personalized_gpt-3.5-instruct-chat_500.pkl

# [0401] obtain indices for gpt-4
python obtain_indices.py --model gpt-4 --mode personation --T_list 1.2 --P_list 1.0
-> indices_dict_goodreads_personation_gpt-4-chat_500.pkl
python obtain_indices.py --model gpt-4 --mode personalized --T_list 1.2 --P_list 1.0
-> indices_dict_goodreads_personalized_gpt-4-chat_500.pkl

# [0401] obtain indices segmented (20,25,30,35,40,45,50)
python obtain_indices_segmented.py
-> indices_dict_segment_goodreads_personalized_llama-2-13b-chat_500.pkl
----------------------------------------------------------------------------------------------------------------
# [0401] obtain mean valid length (right after extracting indices)

python obtain_mean_valid_length.py;\
python obtain_mean_valid_length.py --model vicuna-13b;\
python obtain_mean_valid_length.py --mode personation;\
python obtain_mean_valid_length.py --model vicuna-13b --mode personation;\
->
mean_len_goodreads_personalized_llama-2-13b-chat_500.npy
mean_len_goodreads_personalized_vicuna-13b-chat_500.npy
mean_len_goodreads_personation_llama-2-13b-chat_500.npy
mean_len_goodreads_personation_vicuna-13b-chat_500.npy

python obtain_mean_valid_length.py --chat nonchat --mode personation --T_list 1.2
python obtain_mean_valid_length.py --chat nonchat --mode personalized --T_list 1.2
-> 
mean_len_goodreads_personation_llama-2-13b-nonchat_500.npy
mean_len_goodreads_personalized_llama-2-13b-nonchat_500.npy

python obtain_mean_valid_length.py --model gpt-4 --mode personation --T_list 1.2 --P_list 1.0
python obtain_mean_valid_length.py --model gpt-4 --mode personalized --T_list 1.2 --P_list 1.0
-> mean_len_goodreads_personation_gpt-4-chat_500.npy
-> mean_len_goodreads_personalized_gpt-4-chat_500.npy

python print_mean_valid_length.py
python print_mean_valid_length_nonchat.py
# 9.95,9.92,9.88,9.83,9.81,9.71,9.64,9.42

# [0402] obtain mean valid length for decay (1.2, 50)
python obtain_mean_valid_length_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50
python obtain_mean_valid_length_decay.py --model llama-2-13b --mode personation -et 1.2 -p 50
python obtain_mean_valid_length_decay.py --model vicuna-13b --mode personalized -et 1.2 -p 50
python obtain_mean_valid_length_decay.py --model vicuna-13b --mode personation -et 1.2 -p 50
-> 
mean_len_goodreads_decay-1.2-50_personalized_llama-2-13b-chat_500.npy
mean_len_goodreads_decay-1.2-50_personation_llama-2-13b-chat_500.npy
mean_len_goodreads_decay-1.2-50_personalized_vicuna-13b-chat_500.npy
mean_len_goodreads_decay-1.2-50_personation_vicuna-13b-chat_500.npy

python print_mean_valid_length_decay.py
----------------------------------------------------------------------------------------------------------------
# vicuna-13b personation 5-9
# [done]
at 4;\
export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.90 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.95 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.98 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.90 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.95 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.98 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.90 50;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.95 50;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.98 50;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.90 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.95 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.98 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_5-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 1.00 50;\

----------------------------------------------------------------------------------------------------------------
# llama-2-13b personation 0-4
# [TODO: running on madison]
export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.90 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.95 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 0.98 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.0 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.90 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.95 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 0.98 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.8 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.90 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.95 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 0.98 50;\
bash run_goodreads_gen_personation_multiple_0-4.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 0.5 1.00 50;\

# temp=1.2 has been run before


----------------------------------------------------------------------------------------------------------------
# vicuna-13b personation 0-4
# [TODO: to run]

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.90 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.95 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 0.98 50;\
CUDA_VISIBLE_DEVICES=0 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.0 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.90 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.95 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 0.98 50;\
CUDA_VISIBLE_DEVICES=1 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.8 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.90 50;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.95 50;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 0.98 50;\
CUDA_VISIBLE_DEVICES=2 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 0.5 1.00 50;\

export TRANSFORMERS_CACHE=../cache/;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.90 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.95 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 0.98 50;\
CUDA_VISIBLE_DEVICES=3 bash run_goodreads_gen_personation_multiple_0-4.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.2 1.00 50;\

----------------------------------------------------------------------------------------------------------------
# temperature decay - personalized [running on madison]
# TODO: remind one important TODO at line 1076

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 1.00 50 0 9;\

----------------------------------------------------------------------------------------------------------------
# temperature decay - personation [running on toronto]

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 1.00 50 0 9;\

----------------------------------------------------------------------------------------------------------------
# temperature = 1.5 [running]

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.90 50 0 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.95 50 0 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.98 50 0 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.90 50;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.95 50;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.98 50;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 1.00 50;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_multiple_0.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b 1.5 0.90 50;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personation_multiple_2.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.90 50;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_multiple_8.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.90 50;\

export TRANSFORMERS_CACHE=../cache/huggingface/;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.90 50 0 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.95 50 0 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.98 50 0 9;\
bash run_goodreads_gen_personalized_multiple.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.90 50;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.95 50;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 0.98 50;\
bash run_goodreads_gen_personation_multiple_0-9.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b 1.5 1.00 50;\


----------------------------------------------------------------------------------------------------------------
# temperature decay (1.2, 50) - personalized & personation [running on toronto]

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 1.00 50 0 9;\


export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 1.00 50 0 9;\

----------------------------------------------------------------------------------------------------------------
# temperature decay (1.5, 50) - personalized & personation [running on madison]

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b linear 1.00 50 0 9;\


export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=1;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    meta-llama/Llama-2-13b-chat-hf llama-2-13b exponential 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b linear 1.00 50 0 9;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=0;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personation_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 1.00 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.90 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.95 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 0.98 50 0 9;\
bash run_goodreads_gen_personalized_decay_multiple_config_1.5.sh \
    lmsys/vicuna-13b-v1.5 vicuna-13b exponential 1.00 50 0 9;\

----------------------------------------------------------------------------------------------------------------
# [0403] non-chat generation

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=2;\
bash run_goodreads_gen_nonchat_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 0.90 50;\
bash run_goodreads_gen_nonchat_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 0.95 50;\
bash run_goodreads_gen_nonchat_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 0.98 50;\
bash run_goodreads_gen_nonchat_personation_multiple_0-9.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 1.00 50;\

export TRANSFORMERS_CACHE=../cache/huggingface;\
export CUDA_VISIBLE_DEVICES=3;\
bash run_goodreads_gen_nonchat_personalized_multiple.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 0.90 50 0 9;\
bash run_goodreads_gen_nonchat_personalized_multiple.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 0.95 50 0 9;\
bash run_goodreads_gen_nonchat_personalized_multiple.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 0.98 50 0 9;\
bash run_goodreads_gen_nonchat_personalized_multiple.sh \
    meta-llama/Llama-2-13b-hf llama-2-13b 1.2 1.00 50 0 9;\



----------------------------------------------------------------------------------------------------------------
# [0324] sentiment classification for gen (HF)

python sentiment_goodreads_new.py --model llama-2-13b --mode personalized
python sentiment_goodreads_new.py --model llama-2-13b --mode personation -d 1
python sentiment_goodreads_new.py --model vicuna-13b --mode personalized -d 2
python sentiment_goodreads_new.py --model vicuna-13b --mode personation -d 3

# [0405] sentiment classification for nonchat
python sentiment_goodreads_new.py --chat nonchat --model llama-2-13b --mode personalized -d 2 --T_list 1.2
python sentiment_goodreads_new.py --chat nonchat --model llama-2-13b --mode personation -d 1 --T_list 1.2

# [0325] sentiment classification for gen (HF, decay)

python sentiment_goodreads_decay.py --model llama-2-13b --mode personalized
python sentiment_goodreads_decay.py --model llama-2-13b --mode personation -d 1
python sentiment_goodreads_decay.py --model vicuna-13b --mode personalized -d 2
python sentiment_goodreads_decay.py --model vicuna-13b --mode personation -d 3

# [0402] sentiment classification for gen (HF, decay, 1.2, 50)
python sentiment_goodreads_decay.py --model llama-2-13b --mode personalized -d 1 -et 1.2 -p 50
python sentiment_goodreads_decay.py --model llama-2-13b --mode personation -d 1 -et 1.2 -p 50
python sentiment_goodreads_decay.py --model vicuna-13b --mode personalized -d 2 -et 1.2 -p 50
python sentiment_goodreads_decay.py --model vicuna-13b --mode personation -d 3 -et 1.2 -p 50

# [0325] sentiment classification for gen (gpt-3.5-instruct)

python sentiment_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized
python sentiment_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation -d 1

# [0331] sentiment classification for gen (HF, temp=1.5)
python sentiment_goodreads_new.py --model llama-2-13b --mode personalized -d 0
python sentiment_goodreads_new.py --model llama-2-13b --mode personation -d 1
python sentiment_goodreads_new.py --model vicuna-13b --mode personalized -d 2
python sentiment_goodreads_new.py --model vicuna-13b --mode personation -d 3

# [0401] sentiment classification for gen (gpt-4)

python sentiment_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personalized -d 0 --T_list 1.2 --P_list 1.0
python sentiment_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personation -d 1 --T_list 1.2 --P_list 1.0

----------------------------------------------------------------------------------------------------------------
# [0324] sentiment classification for src

python sentiment_goodreads_src.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en_10/ \
    --output_file ../SafeNLP/results_sentiment/sentiment_goodreads_src_grouped_reviews_long_sub_en_10_sentiment.json

----------------------------------------------------------------------------------------------------------------
# [0324] sentiment summarization for gen [0331 update]
python analyze_sentiment_goodreads.py --model llama-2-13b --mode personalized;\
python analyze_sentiment_goodreads.py --model llama-2-13b --mode personation;\
python analyze_sentiment_goodreads.py --model vicuna-13b --mode personalized;\
python analyze_sentiment_goodreads.py --model vicuna-13b --mode personation;\

python analyze_sentiment_goodreads_src.py \
    --input_file ../SafeNLP/results_sentiment/sentiment_goodreads_src_grouped_reviews_long_sub_en_10_sentiment.json \
    --out_file analysis_results_sentiment/summary_sentiment_goodreads_src_grouped_reviews_long_sub_en_10.pkl

# [0405] sentiment summarization for nonchat
python analyze_sentiment_goodreads.py --model llama-2-13b --mode personalized --chat nonchat --T_list 1.2;\
python analyze_sentiment_goodreads.py --model llama-2-13b --mode personation --chat nonchat --T_list 1.2;\

# [0325] sentiment summarization for gen (decay)
python analyze_sentiment_goodreads_decay.py --model llama-2-13b --mode personalized;\
python analyze_sentiment_goodreads_decay.py --model llama-2-13b --mode personation;\
python analyze_sentiment_goodreads_decay.py --model vicuna-13b --mode personalized;\
python analyze_sentiment_goodreads_decay.py --model vicuna-13b --mode personation

# [0402] sentiment summarization for gen (decay, 1.2, 50)
python analyze_sentiment_goodreads_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50;\
python analyze_sentiment_goodreads_decay.py --model llama-2-13b --mode personation -et 1.2 -p 50;\
python analyze_sentiment_goodreads_decay.py --model vicuna-13b --mode personalized -et 1.2 -p 50;\
python analyze_sentiment_goodreads_decay.py --model vicuna-13b --mode personation -et 1.2 -p 50;\

# [0325] sentiment summarization for gen (gpt-3.5-instruct)
python analyze_sentiment_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized
python analyze_sentiment_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation

# [0401] sentiment summarization for gen (gpt-4)
python analyze_sentiment_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personalized --T_list 1.2 --P_list 1.0
python analyze_sentiment_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personation --T_list 1.2 --P_list 1.0

----------------------------------------------------------------------------------------------------------------
# [0324] topic classification for gen (HF) [0401 update]

python bertopic_goodreads_new.py --model llama-2-13b --mode personalized -d 0
python bertopic_goodreads_new.py --model llama-2-13b --mode personation -d 1
python bertopic_goodreads_new.py --model vicuna-13b --mode personalized -d 2
python bertopic_goodreads_new.py --model vicuna-13b --mode personation -d 3

# for src
python bertopic_goodreads_src.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en_10/ \
    --output_file ../SafeNLP/results_topic/topic_goodreads_src_grouped_reviews_long_sub_en_10.json

# [0405] topic classification for nonchat
python bertopic_goodreads_new.py --model llama-2-13b --mode personalized -d 0 --chat nonchat --T_list 1.2
python bertopic_goodreads_new.py --model llama-2-13b --mode personation -d 1 --chat nonchat --T_list 1.2

# [0325] topic classification for gen (HF, decay)
python bertopic_goodreads_decay.py --model llama-2-13b --mode personalized
python bertopic_goodreads_decay.py --model llama-2-13b --mode personation -d 1
python bertopic_goodreads_decay.py --model vicuna-13b --mode personalized -d 2
python bertopic_goodreads_decay.py --model vicuna-13b --mode personation -d 3

# [0402] topic classification for gen (HF, decay, 1.2, 50)
python bertopic_goodreads_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50
python bertopic_goodreads_decay.py --model llama-2-13b --mode personation -d 1 -et 1.2 -p 50
python bertopic_goodreads_decay.py --model vicuna-13b --mode personalized -d 2 -et 1.2 -p 50
python bertopic_goodreads_decay.py --model vicuna-13b --mode personation -d 3 -et 1.2 -p 50

# [0325] topic classification for gen (gpt-3.5-instruct)

python bertopic_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized -d 2
python bertopic_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation -d 3

# [0401] topic classification for gen (gpt-4)

python bertopic_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personalized -d 0 --T_list 1.2 --P_list 1.0
python bertopic_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personation -d 1 --T_list 1.2 --P_list 1.0

----------------------------------------------------------------------------------------------------------------
# [0324] topic summarization for gen [0401 update]
python analyze_topic_goodreads.py --model llama-2-13b --mode personalized
python analyze_topic_goodreads.py --model llama-2-13b --mode personation
python analyze_topic_goodreads.py --model vicuna-13b --mode personalized
python analyze_topic_goodreads.py --model vicuna-13b --mode personation

python analyze_topic_goodreads_src.py \
    --input_file ../SafeNLP/results_topic/topic_goodreads_src_grouped_reviews_long_sub_en_10.json \
    --out_file analysis_results_topic/summary_topic_goodreads_src_grouped_reviews_long_sub_en_10.pkl

# [0405] topic summarization for nonchat
python analyze_topic_goodreads.py --model llama-2-13b --mode personalized --chat nonchat --T_list 1.2
python analyze_topic_goodreads.py --model llama-2-13b --mode personation --chat nonchat --T_list 1.2

# [0325] topic summarization for gen (decay)
python analyze_topic_goodreads_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50
python analyze_topic_goodreads_decay.py --model llama-2-13b --mode personation -et 1.2 -p 50
python analyze_topic_goodreads_decay.py --model vicuna-13b --mode personalized -et 1.2 -p 50
python analyze_topic_goodreads_decay.py --model vicuna-13b --mode personation -et 1.2 -p 50
->

analysis_results_topic/summary_topic_goodreads_decay-1.2-50_personation_llama-2-13b-chat_500.pkl
analysis_results_topic/summary_topic_goodreads_decay-1.2-50_personalized_vicuna-13b-chat_500.pkl
analysis_results_topic/summary_topic_goodreads_decay-1.2-50_personation_vicuna-13b-chat_500.pkl

# [0325] topic summarization for gen (gpt-3.5-instruct)
python analyze_topic_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized
python analyze_topic_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation

# [0401] topic summarization for gen (gpt-4)
python analyze_topic_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personalized --T_list 1.2 --P_list 1.0
python analyze_topic_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personation --T_list 1.2 --P_list 1.0
----------------------------------------------------------------------------------------------------------------
# [0325] topic distr

python analyze_topic_distr_goodreads.py  --model llama-2-13b --mode personalized
python analyze_topic_distr_goodreads.py --model llama-2-13b --mode personation
python analyze_topic_distr_goodreads.py --model vicuna-13b --mode personalized
python analyze_topic_distr_goodreads.py --model vicuna-13b --mode personation

python analyze_topic_distr_goodreads_src.py \
    --input_file ../SafeNLP/results_topic/topic_goodreads_src_grouped_reviews_long_sub_en_10.json \
    --out_file analysis_results_topic_distr/summary_topic_distr_goodreads_src_grouped_reviews_long_sub_en_10.pkl

# [0405] topic distr for nonchat
python analyze_topic_distr_goodreads.py --model llama-2-13b --mode personalized --chat nonchat --T_list 1.2
python analyze_topic_distr_goodreads.py --model llama-2-13b --mode personation --chat nonchat --T_list 1.2

# [0325] topic distr for decay

# [0325] topic distr for gpt-3.5-instruct
python analyze_topic_distr_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized
python analyze_topic_distr_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation

# [0401] topic distr for gpt-4
python analyze_topic_distr_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personalized --T_list 1.2 --P_list 1.0
python analyze_topic_distr_goodreads_gpt-3.5-instruct.py --model gpt-4 --mode personation --T_list 1.2 --P_list 1.0

----------------------------------------------------------------------------------------------------------------
# [0325] wordfreq

python word_freq_goodreads.py --model llama-2-13b --mode personalized;\
python word_freq_goodreads.py --model llama-2-13b --mode personation;\
python word_freq_goodreads.py --model vicuna-13b --mode personalized;\
python word_freq_goodreads.py --model vicuna-13b --mode personation;\

python wordfreq_goodreads_src.py \
    --input_folder ../review_data/goodreads/grouped_reviews_long_sub_en_10/ \
    --output_file ../SafeNLP/results_wordfreq/wordfreq_goodreads_src_grouped_reviews_long_sub_en_10.json

# [0325] wordfreq gpt-3.5-instruct
python word_freq_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized;\
python word_freq_goodreads_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation

# 


----------------------------------------------------------------------------------------------------------------
# [0325] wordfreq summarization

python analyze_wordfreq.py --model llama-2-13b --mode personalized
python analyze_wordfreq.py --model llama-2-13b --mode personation;\
python analyze_wordfreq.py --model vicuna-13b --mode personalized;\
python analyze_wordfreq.py --model vicuna-13b --mode personation;\

python analyze_wordfreq_src.py \
    --input_file ../SafeNLP/results_wordfreq/wordfreq_goodreads_src_grouped_reviews_long_sub_en_10.json \
    --out_file analysis_results_wordfreq/summary_wordfreq_goodreads_src_grouped_reviews_long_sub_en_10.pkl

# [0325] wordfreq gpt-3.5-instruct
python analyze_wordfreq_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personalized
python analyze_wordfreq_gpt-3.5-instruct.py --model gpt-3.5-instruct --mode personation

----------------------------------------------------------------------------------------------------------------
# [0325] wordfreq analysis

python calc_wordfreq_cossim.py --model llama-2-13b --mode personalized
python calc_wordfreq_cossim.py --model llama-2-13b --mode personation;\
python calc_wordfreq_cossim.py --model vicuna-13b --mode personalized;\
python calc_wordfreq_cossim.py --model vicuna-13b --mode personation;\
python calc_wordfreq_cossim.py --model gpt-3.5-instruct --mode personalized;\
python calc_wordfreq_cossim.py --model gpt-3.5-instruct --mode personation;\

python calc_wordfreq_entropy.py --model llama-2-13b --mode personalized;\
python calc_wordfreq_entropy.py --model llama-2-13b --mode personation;\
python calc_wordfreq_entropy.py --model vicuna-13b --mode personalized;\
python calc_wordfreq_entropy.py --model vicuna-13b --mode personation;\
python calc_wordfreq_entropy.py --model gpt-3.5-instruct --mode personalized;\
python calc_wordfreq_entropy.py --model gpt-3.5-instruct --mode personation;\

python calc_wordfreq_entropy_src.py
-> entropy = 9.99

python calc_wordfreq_count.py --model llama-2-13b --mode personalized;\
python calc_wordfreq_count.py --model llama-2-13b --mode personation;\
python calc_wordfreq_count.py --model vicuna-13b --mode personalized;\
python calc_wordfreq_count.py --model vicuna-13b --mode personation;\
python calc_wordfreq_count.py --model gpt-3.5-instruct --mode personalized;\
python calc_wordfreq_count.py --model gpt-3.5-instruct --mode personation;\

python calc_wordfreq_count_src.py
-> count = 85334

----------------------------------------------------------------------------------------------------------------
# [0326] wordfreq clustering

python perform_wordfreq_cluster.py --model llama-2-13b --mode personalized;\

----------------------------------------------------------------------------------------------------------------
# [0401] high perplexity samples

python print_high_ppl_samples.py > samples_at_diff_perplexity_intervals.txt

