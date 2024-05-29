import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--input_id', type=str, default='codeonly')
args = parser.parse_args()

stats = np.load(f'run_judge_stats/stats_{args.input_id}.npz')

outcomes = stats['outcomes']
outcomes_cnt = outcomes.sum(axis=2)
outcomes_acc = np.where(outcomes_cnt==10, 1, 0)

indices = np.where(outcomes_acc.sum(axis=-1)<20)[0]

print('err indices', indices)
print('outcomes_acc', outcomes_acc.shape)

full_outcomes = []
acc_overall = []
cnt = 0
for i in range(100):
    acc_overall.append(outcomes_acc[i].mean())

assert len(acc_overall) == 100
np.save(f'acc_train_gpt4_{args.input_id}.npy', acc_overall)