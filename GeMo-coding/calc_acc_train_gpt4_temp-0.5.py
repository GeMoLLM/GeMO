import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--input_id', type=str, default='codeonly')
args = parser.parse_args()

stats_1 = np.load(f'run_judge_stats/stats_codeonly_temp-0.5.npz')
stats_2 = np.load('run_judge_stats/stats_codeonly_temp-0.5_sel_temp-0.5.npz')

outcomes_1 = stats_1['outcomes']
outcomes_cnt_1 = outcomes_1.sum(axis=2)
outcomes_acc_1 = np.where(outcomes_cnt_1==10, 1, 0)

indices = np.where(outcomes_acc_1.sum(axis=-1)<20)[0]
print('err indices', indices)

outcomes_2 = stats_2['outcomes']
outcomes_cnt_2 = outcomes_2.sum(axis=2)
outcomes_acc_2 = np.where(outcomes_cnt_2==10, 1, 0)

print('outcomes_acc_1', outcomes_acc_1.shape)
print('outcomes_acc_2', outcomes_acc_2.shape)

full_outcomes = []
acc_overall = []
cnt = 0
for i in range(100):
    if i in indices:
        acc_overall.append(outcomes_acc_2[cnt].mean())
        cnt += 1
    else:
        acc_overall.append(outcomes_acc_1[i].mean())

assert cnt == 5
assert len(acc_overall) == 100
np.save('acc_train_gpt4_codeonly_temp-0.5.npy', acc_overall)