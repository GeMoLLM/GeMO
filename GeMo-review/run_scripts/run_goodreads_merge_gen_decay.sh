set -x
for i in $(seq $6 $7);
do
    python merge.py --prefix goodreads_completions_$1_$2-chat_500_decay-$3_p-$4_k-$5-$i --n 5;\
done