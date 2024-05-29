import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_prefix', type=str, default='valid_solution')
parser.add_argument('--outfile_id', type=str, default='gpt4_gen_1')
args = parser.parse_args()

indexs = np.load('./indexs.npy')
indices = np.where(indexs == 'A')[0]
cnt = len(indices)

folder = 'judge_output'
arr = np.zeros((cnt, 3), dtype=int)
for ii, i in enumerate(indices):
    filename = f'{folder}/{args.input_prefix}_{i}.py'
    lines = open(filename).readlines()
    compile = int(lines[0].strip())
    n_pass = int(lines[1].strip())
    runtime = int(lines[2].strip())
    arr[ii] = (compile, n_pass, runtime)
    print(compile, n_pass, runtime)
    
np.save(f'judge_output_{args.outfile_id}.npy', arr)