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
args = parser.parse_args()

T_list = [float(x) for x in args.T_list.split(',')]
P_list = [float(x) for x in args.P_list.split(',')]
    
folder = 'results_topic'
out_folder = 'analysis_results_topic_distr'
os.makedirs(out_folder, exist_ok=True)
result = {T: {P: {} for P in P_list} for T in T_list}

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))

data_src = json.load(open('../SafeNLP/results_topic/topic_goodreads_src_grouped_reviews_long_sub_en_10.json'))
books = set(list(data_src.keys()))
print(len(books))

for T in T_list:
    for P in P_list:
        filepath = f'{folder}/topic_goodreads_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.2f}.json' \
            if args.model == 'gpt-3.5-instruct' \
                else f'{folder}/topic_goodreads_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.1f}.json'
        data = json.load(open(filepath))
        topics = []
        for k, v in data.items():
            if k in books:
                topics += v
        result[T][P] = np.unique(topics, return_counts=True)

pickle.dump(result, open(f'{out_folder}/summary_topic_distr_goodreads_{args.mode}_{args.model}-chat_500.pkl', 'wb'))
