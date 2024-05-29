import argparse
import spacy
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import json
import pickle
from analyze_utils import read_in_texts, get_wordfreq

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
args = parser.parse_args()

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

# prepare the indices_dict
indices_dict_path = f'indices_dict_goodreads_{args.mode}_{args.model}-chat_500.pkl'
indices_dict = pickle.load(open(indices_dict_path, 'rb'))
print('load in indices_dict!')

out_folder = 'results_wordfreq'
os.makedirs(out_folder, exist_ok=True)

data_src = json.load(open('../SafeNLP/results_topic/topic_goodreads_src_grouped_reviews_long_sub_en_10.json'))
books = set(list(data_src.keys()))
print(len(books))

for T in T_list:
    for P in P_list:
        input_folder = f'folder_goodreads_completions_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.2f}'

        files = os.listdir(input_folder)
        d_wordfreq = {}
        texts_all = []
        for file in tqdm(files):
            if file in books:
                file_path = osp.join(input_folder, file)
                texts = read_in_texts(
                    file_path,  
                    indices_dict[T][P][file.replace('.jsonl', '')])
                texts_all += texts

        wordfreq = get_wordfreq(texts_all)
        
        out_file = f'{out_folder}/wordfreq_goodreads_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P:.2f}.json'
        print(out_file)
        json.dump(wordfreq, open(out_file, 'w'))