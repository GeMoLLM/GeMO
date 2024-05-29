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

decay_list = ['linear', 'exponential']
P_list = [0.90, 0.95, 0.98, 1.00]

def get_mean(l):
    return np.mean(l)

def get_std(l):
    return np.std(l)

def get_entropy(l):
    _, count = np.unique(l, return_counts=True)
    if len(count) == 1:
        return 0
    return -np.sum(count/len(l) * np.log2(count/len(l)))
    
def transform(l):
    unique = np.unique(l)
    if unique[0] == 'POSITIVE' or unique[0] == 'NEGATIVE':
        return [1 if x == 'POSITIVE' else 0 for x in l]
    assert unique[0] == 0 or unique[0] == 1, f'l contains {unique}'
    return l

folder = 'results_sentiment'
out_folder = 'analysis_results_sentiment'
os.makedirs(out_folder, exist_ok=True)
result = {decay: {P: {} for P in P_list} for decay in decay_list}

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))

for decay in decay_list:
    for P in P_list:
        filepath = f'{folder}/sentiment_goodreads_{args.mode}_{args.model}-chat_500_decay-{decay}_p-{P:.2f}_k-50.json' \
            if args.end_temperature == 1.0 and args.period == 20 \
                else f'{folder}/sentiment_goodreads_{args.mode}_{args.model}-chat_500_decay-{decay}-{args.end_temperature}-{args.period}_p-{P:.2f}_k-50.json'
        data = json.load(open(filepath))
        for k, v in data.items():
            l = transform(v)
            result[decay][P][k] = {
                'mean': get_mean(l),
                'std': get_std(l),
                'entropy': get_entropy(l)
            }

final_results = {book: {'mean': np.zeros((4,4)),
                        'std':  np.zeros((4,4)),
                        'entropy': np.zeros((4,4))} for book in book_maps.keys()}
for book in book_maps.keys():
    book_file = book + '.jsonl'
    for ti, decay in enumerate(decay_list):
        for pi, P in enumerate(P_list):
            final_results[book]['mean'][ti][pi] = result[decay][P][book_file]['mean']
            final_results[book]['std'][ti][pi] = result[decay][P][book_file]['std']
            final_results[book]['entropy'][ti][pi] = result[decay][P][book_file]['entropy']

out_file = f'{out_folder}/summary_sentiment_goodreads_decay_{args.mode}_{args.model}-chat_500.pkl' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'{out_folder}/summary_sentiment_goodreads_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.pkl'
pickle.dump(final_results, open(out_file, 'wb'))
