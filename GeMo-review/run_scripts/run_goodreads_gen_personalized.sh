set -x
python generation_goodreads.py --temperature $3 --model_path $1 --top_p $4 --top_k $5 --output_path goodreads_completions_personalized_$2-chat_500_temp-$3_p-$4_k-$5-0
python generation_goodreads.py --temperature $3 --model_path $1 --top_p $4 --top_k $5 --output_path goodreads_completions_personalized_$2-chat_500_temp-$3_p-$4_k-$5-1
python generation_goodreads.py --temperature $3 --model_path $1 --top_p $4 --top_k $5 --output_path goodreads_completions_personalized_$2-chat_500_temp-$3_p-$4_k-$5-2
python generation_goodreads.py --temperature $3 --model_path $1 --top_p $4 --top_k $5 --output_path goodreads_completions_personalized_$2-chat_500_temp-$3_p-$4_k-$5-3
python generation_goodreads.py --temperature $3 --model_path $1 --top_p $4 --top_k $5 --output_path goodreads_completions_personalized_$2-chat_500_temp-$3_p-$4_k-$5-4
