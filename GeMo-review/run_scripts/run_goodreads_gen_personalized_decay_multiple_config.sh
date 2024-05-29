set -x
for i in $(seq $6 $7);
do
    python generation_goodreads_decay.py \
        --model_path $1 \
        --top_p $4 \
        --top_k $5 \
        --decay $3 \
        --end_temperature 1.2 \
        --period 50 \
        --output_path goodreads_completions_personalized_$2-chat_500_decay-$3-1.2-50_p-$4_k-$5-$i;
done