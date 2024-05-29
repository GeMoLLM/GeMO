import pickle
import argparse
import numpy as np
import os.path as osp
import os
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
args = parser.parse_args()

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

folder = 'results_wordfreq'
out_folder = 'analysis_results_wordfreq'
os.makedirs(out_folder, exist_ok=True)
result = {T: {P: {} for P in P_list} for T in T_list}

data_src = json.load(open('../SafeNLP/results_topic/topic_goodreads_src_grouped_reviews_long_sub_en_10.json'))
books = set(list(data_src.keys()))
print(len(books))

for T in T_list:
    for P in P_list:
        filepath = f'{folder}/wordfreq_goodreads_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.2f}.json'
        result[T][P] = json.load(open(filepath))

pickle.dump(result, open(f'{out_folder}/summary_wordfreq_goodreads_{args.mode}_{args.model}-chat_500.pkl', 'wb'))