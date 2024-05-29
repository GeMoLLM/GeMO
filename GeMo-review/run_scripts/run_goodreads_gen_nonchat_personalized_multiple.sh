set -x
for i in $(seq $6 $7);
do
    python generation_goodreads.py \
        -nc \
        --temperature $3 \
        --model_path $1 \
        --top_p $4 \
        --top_k $5 \
        --output_path goodreads_completions_personalized_$2-nonchat_500_temp-$3_p-$4_k-$5-$i;
done