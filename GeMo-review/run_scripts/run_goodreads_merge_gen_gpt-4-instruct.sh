set -x
for i in $(seq $4 $5);
do
    python merge.py --prefix goodreads_completions_$1_gpt-4-chat_500_temp-$2-p-$3-$i --n 75;\
done