from transformers import pipeline
import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--device', '-d', type=int, default=0)
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

decay_list = ['linear', 'exponential']
P_list = [0.90, 0.95, 0.98, 1.00]
os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# prepare the indices_dict
indices_dict_path = f'indices_dict_goodreads_decay_{args.mode}_{args.model}-chat_500.pkl' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'indices_dict_goodreads_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.pkl'
indices_dict = pickle.load(open(indices_dict_path, 'rb'))
print('load in indices_dict!')

# load in the sentiment classifier
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                              max_length=512,
                              truncation=True,
                              device=0)
print('load in sentiment classifier!')

def read_in_texts(file_path, indices):
    texts = []
    with open(file_path) as f:
        for line in f:
            texts.append(json.loads(line))
    return [texts[i] for i in indices]

out_folder = 'results_sentiment'
os.makedirs(out_folder, exist_ok=True)
for decay in decay_list:
    for P in P_list:
        input_folder = f'folder_goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{decay}_p-{P:.2f}_k-50' \
            if args.end_temperature == 1.0 and args.period == 20 \
                else f'folder_goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{decay}-{args.end_temperature}-{args.period}_p-{P:.2f}_k-50'
                
        out_file = f'{out_folder}/sentiment_goodreads_{args.mode}_{args.model}-chat_500_decay-{decay}_p-{P:.2f}_k-50.json' \
            if args.end_temperature == 1.0 and args.period == 20 \
                else f'{out_folder}/sentiment_goodreads_{args.mode}_{args.model}-chat_500_decay-{decay}-{args.end_temperature}-{args.period}_p-{P:.2f}_k-50.json'
                
        if osp.exists(out_file):
            print(f'{out_file} already exists. skipping...')
            continue

        files = os.listdir(input_folder)
        d_sent = {}
        for file in tqdm(files):
            file_path = osp.join(input_folder, file)
            texts = read_in_texts(
                file_path,  
                indices_dict[decay][P][file.replace('.jsonl', '')])
            out = sentiment_pipeline(texts)
            out = [x['label'] for x in out]

            d_sent[file] = out
            
        print(out_file)
        json.dump(d_sent, open(out_file, 'w'))