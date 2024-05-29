set -x
for i in 1.2; do
  for j in 0.90 0.95 0.98 1.00; do
    bash run_goodreads_merge_gen_nonchat.sh $1 $2 $i $j 50 $3 $4
  done
done