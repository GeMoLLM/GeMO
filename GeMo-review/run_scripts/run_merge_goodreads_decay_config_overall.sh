set -x
for i in linear exponential; do
  for j in 0.90 0.95 0.98 1.00; do
    bash run_goodreads_merge_gen_decay_config.sh $1 $2 $i $j 50 $3 $4 $5 $6
  done
done