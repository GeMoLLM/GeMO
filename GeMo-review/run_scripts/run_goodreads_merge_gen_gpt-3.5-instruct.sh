set -x
for i in $(seq $4 $5);
do
    python merge.py --prefix goodreads_completions_$1_gpt-3.5-instruct-chat_500_temp-$2-p-$3-$i --n 5;\
done