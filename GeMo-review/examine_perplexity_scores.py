import argparse
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--T_list', type=str, default='0.5,0.8,1.0,1.2,1.5')
parser.add_argument('--P_list', type=str, default='0.90,0.95,0.98,1.00')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--n_gen', type=int, default=10)
parser.add_argument('--chat', type=str, default='chat')
args = parser.parse_args()

T_list = args.T_list.split(',')
P_list = args.P_list.split(',')

folder_path = 'perplexity_scores'

result = np.zeros((len(T_list), len(P_list), args.n_gen, 750))
for ti, T in enumerate(T_list):
    for pi, P in enumerate(P_list):
        for i in range(args.n_gen):
            data_path = f'perplexity_goodreads_completions_{args.mode}_{args.model}-{args.chat}_500_temp-{T}_p-{P}_k-50-{i}_merged.npy'
            result[ti][pi][i] = np.load(osp.join(folder_path, data_path))
                
out_path = osp.join(folder_path, f'perplexity_goodreads_completions_{args.mode}_{args.model}-{args.chat}_500.npy')
print(out_path)
np.save(out_path, result)