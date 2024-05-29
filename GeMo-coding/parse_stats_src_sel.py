import numpy as np
import os.path as osp

# Sample input as a string
folder = 'run_judge_stats'

import argparse
parser = argparse.ArgumentParser(description='Parse the stats from the run_judge.sh')
parser.add_argument('--n_probs', type=int, default=16)
parser.add_argument('--n_src', type=int, default=25)
args = parser.parse_args()

input_filename = './codeforces_A_file_paths_sel.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:args.n_probs]

outcomes = np.zeros((args.n_probs, args.n_src, 10), dtype=int)
runtimes = np.zeros((args.n_probs, args.n_src, 10), dtype=int)
memories = np.zeros((args.n_probs, args.n_src, 10), dtype=int)

for fi, file_path in enumerate(file_paths):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    for i in range(args.n_src):
        with open(osp.join(folder, f'src_stats_{fileid}_{pid}_{i}.txt'), 'r') as file:
            input_str = file.read()

        # Split the input string into lines
        lines = input_str.strip().split('\n')

        # Initialize arrays
        cur_outcomes = np.zeros(10, dtype=int)
        cur_runtimes = np.zeros(10, dtype=int)
        cur_memories = np.zeros(10, dtype=int)

        # Parse each line
        for j, line in enumerate(lines):
            # print(fileid, pid, i, j, line)
            parts = line.split()
            # Get the test case number
            case_num = int(parts[2])
            assert case_num == j, f"case_num={case_num}, i={i}, fileid={fileid}, pid={pid}"

            # Parse the outcome
            if parts[3] == 'passed':
                cur_outcomes[case_num] = 1
            elif parts[3] == 'failed':
                cur_outcomes[case_num] = 0
            else:  # Timed out
                cur_outcomes[case_num] = -1
                cur_runtimes[case_num] = -1
                cur_memories[case_num] = -1
                continue
            # Parse the runtime and memory
            cur_runtimes[case_num] = int(float(parts[5]))
            cur_memories[case_num] = int(parts[8])

        # Output the arrays
        outcomes[fi, i] = cur_outcomes
        runtimes[fi, i] = cur_runtimes
        memories[fi, i] = cur_memories

# Save the arrays
np.savez(osp.join(folder, f'src_sel_stats.npz'), outcomes=outcomes, runtimes=runtimes, memories=memories)