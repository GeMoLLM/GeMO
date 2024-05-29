import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--input_id', type=str, default='codeonly')
args = parser.parse_args()

data1 = np.load('run_judge_stats/stats_claude_codeonly_temp-0.5_p-1.0.npz')

outcomes1 = data1['outcomes']

print(outcomes1.shape)

intermediate1 = np.where(np.where(outcomes1==1, 1, 0).sum(axis=-1)==10, 1, 0)

stats1 = intermediate1.sum(axis=-1)

print('intermediate1', intermediate1.shape)
print('stats1:', stats1.shape, stats1)

indices = list(np.where(stats1<20)[0])

acc_overall = []
cnt = 0
for i in range(50):
    acc_overall.append(outcomes1[i].mean())

print('cnt', cnt)
# assert cnt == 5
print(len(acc_overall))
assert len(acc_overall) == 50
np.save('acc_train_claude_codeonly_temp-0.5_p-1.0.npy', acc_overall)