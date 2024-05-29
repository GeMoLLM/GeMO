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
parser.add_argument('--T_list', type=str, default='0.5,0.8,1.0,1.2,1.5')
parser.add_argument('--P_list', type=str, default='0.90,0.95,0.98,1.00')
parser.add_argument('--chat', type=str, default='chat')
args = parser.parse_args()

indices = pickle.load(open(f'indices_dict_goodreads_{args.mode}_{args.model}-{args.chat}_500.pkl', 'rb'))

T_list = args.T_list.split(',')
P_list = args.P_list.split(',')

mean_len = np.zeros((len(T_list), len(P_list)), dtype=float)
for ti, T in enumerate(T_list):
    for pi, P in enumerate(P_list):
        mean_len[ti][pi] = np.array([len(x) for x in indices[T][P].values()]).mean()
        
out_file = f'mean_len_goodreads_{args.mode}_{args.model}-{args.chat}_500.npy'
print(out_file)
np.save(out_file, mean_len)