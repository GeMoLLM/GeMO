from transformers import pipeline
import argparse
import spacy
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import json
import umap.umap_ as UMAP
from bertopic import BERTopic

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--typ', type=str, default='src')
args = parser.parse_args()

topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")

files = os.listdir(args.input_folder)
d_topic = {}
for file in tqdm(files):
    file_path = osp.join(args.input_folder, file)
    src_cur = []
    texts = []
    with open(file_path) as f:
        texts = []
        if args.typ == 'src':
            for line in f:
                texts.append(json.loads(line)['review_text'])
        else:
            for line in f:
                texts.append(json.loads(line))
                
        topics, scores = topic_model.transform(texts)
        
    d_topic[file] = list([int(x) for x in topics])

json.dump(d_topic, open(args.output_file, 'w'))