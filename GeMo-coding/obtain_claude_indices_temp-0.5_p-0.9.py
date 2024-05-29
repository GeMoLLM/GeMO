import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

data1 = np.load('run_judge_stats/stats_claude_codeonly_temp-0.5_p-0.9.npz')
data2 = np.load('run_judge_stats/stats_claude_codeonly_temp-0.5_p-0.9_sel_2.npz')

outcomes1 = data1['outcomes']
outcomes2 = data2['outcomes']

print(outcomes1.shape)
print(outcomes2.shape)

intermediate1 = np.where(np.where(outcomes1==1, 1, 0).sum(axis=-1)==10, 1, 0)
intermediate2 = np.where(np.where(outcomes2==1, 1, 0).sum(axis=-1)==10, 1, 0)

stats1 = intermediate1.sum(axis=-1)
stats2 = intermediate2.sum(axis=-1)

print('intermediate1', intermediate1.shape)
print('intermediate2', intermediate2.shape)
print('stats1:', stats1.shape, stats1)
print('stats2:', stats2.shape, stats2)

indices = list(np.where(stats1<20)[0])

all_p = []
for i in range(50):
    if i in indices:
        idx = indices.index(i)
        all_idx = np.where(intermediate2[idx] == 1)[0]
        if len(all_idx) < 20:
            print('! indices here:', 'idx:', idx, 'all_idx:', all_idx, 'len(all_idx):', len(all_idx))
            remain_idx = np.array(list(set(list(range(len(intermediate1[idx])))) - set(all_idx)))
            sub_remain_idx = np.random.choice(remain_idx, 20-len(all_idx), replace=False)
            sub_idx = np.concatenate([all_idx, sub_remain_idx])
        else:
            sub_idx = sorted(np.random.choice(all_idx, 20, replace=False))
    else:
        all_idx = np.where(intermediate1[i] == 1)[0]
        assert len(all_idx) >= 20, f'i={i}, len(all_idx)={len(all_idx)}'
        sub_idx = sorted(np.random.choice(all_idx, 20, replace=False))
    all_p.append(sub_idx)

all_p = np.array(all_p)
print('all_p', all_p.shape)

out_file = 'codeforces_A_gen_claude_temp-0.5_p-0.9_index.npy'
np.save(out_file, all_p)