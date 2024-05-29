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
from analyze_utils import read_in_src_texts

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()

topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")

files = os.listdir(args.input_folder)
d_topic = {}
for file in tqdm(files):
    file_path = osp.join(args.input_folder, file)
    src_cur = []
    texts = read_in_src_texts(file_path)
                
    topics, scores = topic_model.transform(texts)
        
    d_topic[file] = list([int(x) for x in topics])

json.dump(d_topic, open(args.output_file, 'w'))