import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

indices = pickle.load(open(f'indices_dict_goodreads_decay_{args.mode}_{args.model}-chat_500.pkl', 'rb')) \
    if args.end_temperature == 1.0 and args.period == 20 \
        else pickle.load(open(f'indices_dict_goodreads_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.pkl', 'rb'))    

decay_list = ['linear', 'exponential']
P_list = [0.90, 0.95, 0.98, 1.00]

mean_len = np.zeros((len(decay_list), len(P_list)), dtype=float)
for ti, T in enumerate(decay_list):
    for pi, P in enumerate(P_list):
        mean_len[ti][pi] = np.array([len(x) for x in indices[T][P].values()]).mean()
        
out_file = f'mean_len_goodreads_decay_{args.mode}_{args.model}-chat_500.npy' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'mean_len_goodreads_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.npy'
    
print(out_file)
np.save(out_file, mean_len)