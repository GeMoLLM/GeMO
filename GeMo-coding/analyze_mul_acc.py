import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='gpt4_gen_codeonly')
parser.add_argument('--n_gen', type=int, default=5)
args = parser.parse_args()

data_list = []
for i in range(1, args.n_gen+1):
    file = f'judge_output_{args.input_id}_{i}.npy'
    data = np.load(file)[:,1]
    data_list.append(data)
    
data = np.array(data_list)
print(data)

print(data.mean(axis=0))
print(data.std(axis=0))

np.save(f'judge_output_{args.input_id}_overall.npy', data)
