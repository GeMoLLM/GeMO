import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--T_list', type=str, default='0.5,0.8,1.0,1.2')
parser.add_argument('--P_list', type=str, default='0.90,0.95,0.98,1.00')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

T_list = [float(x) for x in args.T_list.split(',')]
P_list = [float(x) for x in args.P_list.split(',')]

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
result = {T: {P: {} for P in P_list} for T in T_list}

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))

for T in T_list:
    for P in P_list:
        filepath = f'{folder}/sentiment_goodreads_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.2f}.json' \
            if args.model == 'gpt-3.5-instruct' \
                else f'{folder}/sentiment_goodreads_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.1f}.json'
        data = json.load(open(filepath))
        for k, v in data.items():
            l = transform(v)
            result[T][P][k] = {
                'mean': get_mean(l),
                'std': get_std(l),
                'entropy': get_entropy(l)
            }

final_results = {book: {'mean': np.zeros((len(T_list), len(P_list))),
                        'std':  np.zeros((len(T_list), len(P_list))),
                        'entropy': np.zeros((len(T_list), len(P_list)))} for book in book_maps.keys()}
for book in book_maps.keys():
    book_file = book + '.jsonl'
    for ti, T in enumerate(T_list):
        for pi, P in enumerate(P_list):
            final_results[book]['mean'][ti][pi] = result[T][P][book_file]['mean']
            final_results[book]['std'][ti][pi] = result[T][P][book_file]['std']
            final_results[book]['entropy'][ti][pi] = result[T][P][book_file]['entropy']

pickle.dump(final_results, open(f'{out_folder}/summary_sentiment_goodreads_{args.mode}_{args.model}-chat_500.pkl', 'wb'))
