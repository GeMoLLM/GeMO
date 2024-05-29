import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data1 = np.load('run_judge_stats/stats_claude_codeonly_temp-1.0_p-0.9.npz')
data2 = np.load('run_judge_stats/stats_claude_codeonly_temp-1.0_p-0.9_sel_17.npz')

outcomes1 = data1['outcomes']
outcomes2 = data2['outcomes']

print(outcomes1.shape)

intermediate1 = np.where(np.where(outcomes1==1, 1, 0).sum(axis=-1)==10, 1, 0)
intermediate2 = np.where(np.where(outcomes2==1, 1, 0).sum(axis=-1)==10, 1, 0)

stats1 = intermediate1.sum(axis=-1)
stats2 = intermediate2.sum(axis=-1)
indices = np.where(stats1>=5)[0]
print(len(indices))

print('intermediate1', intermediate1.shape)
print('intermediate2', intermediate2.shape)
print('stats1:', stats1.shape, stats1)
print('stats2:', stats2.shape, stats2)

indices = np.delete(indices, 1)[:-1]

print('indices:', len(indices), indices)
np.save('claude_all_indices.npy', indices)
indices2 = list(indices[np.where(stats1[indices]<20)[0]])

print('indices2:', indices2, type(indices2))

all_p = []
for i in range(100):
    if i not in indices:
        continue
    if i in indices2:
        idx = indices2.index(i)
        all_idx = np.where(intermediate2[idx] == 1)[0]
        if len(all_idx) < 20:
            print('! indices2 here:', 'idx:', idx, 'all_idx:', all_idx, 'len(all_idx):', len(all_idx))
            remain_idx = np.array(list(set(list(range(len(intermediate2[idx])))) - set(all_idx)))
            sub_remain_idx = np.random.choice(remain_idx, 20-len(all_idx), replace=False)
            sub_idx = np.concatenate([all_idx, sub_remain_idx])
        else:
            sub_idx = sorted(np.random.choice(all_idx, 20, replace=False))
    else:
        all_idx = np.where(intermediate1[i] == 1)[0]
        assert len(all_idx) >= 20, f'i={i}, len(all_idx)={len(all_idx)}'
        sub_idx = sorted(np.random.choice(all_idx, 20, replace=False))
        
    if i == 0:
        print('--------------', sub_idx)
    all_p.append(sub_idx)
    
all_p = np.array(all_p)
print('all_p', all_p.shape)

out_file = 'codeforces_A_gen_claude_temp-1.0_p-0.9_index.npy'
np.save(out_file, all_p)


with open('codeforces_A_file_paths_final.txt', 'r') as f:
    file_paths = f.readlines()
    
sel_file_paths = [file_paths[i] for i in indices]

assert len(sel_file_paths) == 50

with open('codeforces_A_file_paths_claude_final.txt', 'w') as f:
    for line in sel_file_paths:
        f.write(line)