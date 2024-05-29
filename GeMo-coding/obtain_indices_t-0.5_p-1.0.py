import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_id", type=str, default='codeonly_temp-0.5_p-1.0')
parser.add_argument("--idx_fileid", type=str, default='_temp-0.5_p-1.0')
args = parser.parse_args()

stats = np.load(f'run_judge_stats/stats_{args.input_id}.npz')

stats = np.where(stats['outcomes'].sum(axis=-1)==10, 1, 0)
score = stats.sum(axis=-1)
print(score)

assert sum(score>20) == 99

final_p = []
for i in range(100):
    indices = np.where(stats[i] == 1)[0]
    if len(indices) < 20:
        additional_indice = np.random.choice(np.where(stats[i] == 0)[0], 20-len(indices), replace=False)
        indices = sorted(np.concatenate([indices, additional_indice]))
    else:
        indices = sorted(np.random.choice(indices, 20, replace=False))
    final_p.append(indices)
final_p = np.array(final_p)

print(final_p)
print(final_p.shape)
print(final_p[15])

np.save(f'codeforces_A_gen_gpt4{args.idx_fileid}_index.npy', final_p)