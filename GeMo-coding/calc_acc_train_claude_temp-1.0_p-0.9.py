import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--input_id', type=str, default='codeonly')
args = parser.parse_args()

data1 = np.load(f'run_judge_stats/stats_claude_codeonly_temp-1.0_p-0.9.npz')
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
indices2 = list(indices[np.where(stats1[indices]<20)[0]])

acc_overall = []
cnt = 0
for i in range(100):
    if i not in indices:
        continue
    if i in indices2:
        idx = indices2.index(i)
        acc_overall.append(outcomes2[idx].mean())
        cnt += 1
    else:
        acc_overall.append(outcomes1[i].mean())

print('cnt', cnt)
# assert cnt == 5
print(len(acc_overall))
assert len(acc_overall) == 50
np.save('acc_train_claude_codeonly_temp-1.0_p-0.9.npy', acc_overall)