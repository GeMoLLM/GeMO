import argparse
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--P_list', type=str, default='0.90,0.95,0.98,1.00')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--n_gen', type=int, default=10)
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

decay_list = ['linear', 'exponential']
P_list = args.P_list.split(',')

folder_path = 'perplexity_scores'

result = np.zeros((len(decay_list), len(P_list), args.n_gen, 750))
for di, decay in enumerate(decay_list):
    for pi, P in enumerate(P_list):
        for i in range(args.n_gen):
            data_path = f'perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{decay}-{args.end_temperature}-{args.period}_p-{P}_k-50-{i}_merged.npy' \
                if args.end_temperature != 1.0 or args.period != 20 \
                    else f'perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{decay}_p-{P}_k-50-{i}_merged.npy'
            result[di][pi][i] = np.load(osp.join(folder_path, data_path))
                
out_path = osp.join(folder_path, f'perplexity_goodreads_completions_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.npy') \
    if args.end_temperature != 1.0 or args.period != 20 \
        else osp.join(folder_path, f'perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500.npy')
print(out_path)
np.save(out_path, result)