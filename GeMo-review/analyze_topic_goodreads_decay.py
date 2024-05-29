import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import json
import pickle
from metrics import get_entropy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

decay_list = ['linear', 'exponential']
P_list = [0.90, 0.95, 0.98, 1.00]
    
folder = 'results_topic'
out_folder = 'analysis_results_topic'
os.makedirs(out_folder, exist_ok=True)
result = {decay: {P: {} for P in P_list} for decay in decay_list}

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))

for decay in decay_list:
    for P in P_list:
        filepath = f'{folder}/topic_goodreads_{args.mode}_{args.model}-chat_500_decay-{decay}_p-{P:.2f}_k-50.json' \
            if args.end_temperature == 1.0 and args.period == 20 \
                else f'{folder}/topic_goodreads_{args.mode}_{args.model}-chat_500_decay-{decay}-{args.end_temperature}-{args.period}_p-{P:.2f}_k-50.json'
        data = json.load(open(filepath))
        for k, v in data.items():
            if not v:
                result[decay][P][k] = {
                    'entropy': -1
                }

            else:
                result[decay][P][k] = {
                    'entropy': get_entropy(v)
                }

final_results = {book: {'entropy': np.zeros((len(decay_list), len(P_list)))} for book in book_maps.keys()}
for book in book_maps.keys():
    book_file = book + '.jsonl'
    for ti, T in enumerate(decay_list):
        for pi, P in enumerate(P_list):
            final_results[book]['entropy'][ti][pi] = result[T][P][book_file]['entropy']

out_file = f'{out_folder}/summary_topic_goodreads_decay_{args.mode}_{args.model}-chat_500.pkl' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'{out_folder}/summary_topic_goodreads_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.pkl'
print(out_file)
pickle.dump(final_results, open(out_file, 'wb'))
