set -x
for i in $(seq $8 $9);
do
    python merge.py --prefix goodreads_completions_$1_$2-chat_500_decay-$3-$6-$7_p-$4_k-$5-$i --n 5;\
done