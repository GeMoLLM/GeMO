import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--input_id', type=str, default='codeonly')
args = parser.parse_args()

stats = np.load(f'run_judge_stats/stats_{args.input_id}.npz')
outcomes = stats['outcomes']
outcomes_cnt = outcomes.sum(axis=2)
outcomes_acc = np.where(outcomes_cnt==10, 1, 0)

print('outcomes_acc', outcomes_acc.shape)

good_sel_indices = np.load('good_sel_indices.npy')
outcomes_acc_sel = outcomes_acc[good_sel_indices]

out_file = f'acc_train_gpt4_{args.input_id}.npy'
print(out_file)
np.save(out_file, outcomes_acc_sel.mean(axis=1))