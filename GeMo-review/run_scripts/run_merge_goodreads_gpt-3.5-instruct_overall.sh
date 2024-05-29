set -x
for i in 0.5 0.8 1.0 1.2; do
  for j in 0.90 0.95 0.98 1.00; do
    bash run_goodreads_merge_gen_gpt-3.5-instruct.sh $1 $i $j $2 $3
  done
done