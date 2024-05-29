from transformers import pipeline
import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--typ', type=str, default='src')
args = parser.parse_args()

sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                              max_length=512,
                              truncation=True,
                              device=0)

files = os.listdir(args.input_folder)
d_sent = {}
for file in tqdm(files):
    file_path = osp.join(args.input_folder, file)
    src_cur = []
    texts = []
    with open(file_path) as f:
        if args.typ == 'src':
            for line in f:
                texts.append(json.loads(line)['review_text'])
        else:
            for line in f:
                texts.append(json.loads(line))
        out = sentiment_pipeline(texts)
        out = [x['label'] for x in out]

    d_sent[file] = out
    
json.dump(d_sent, open(args.output_file, 'w'))