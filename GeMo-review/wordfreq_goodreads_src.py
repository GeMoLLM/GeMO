import argparse
import spacy
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import json
from analyze_utils import read_in_src_texts, get_wordfreq

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()

files = os.listdir(args.input_folder)
d_topic = {}
texts_all = []
for file in tqdm(files):
    file_path = osp.join(args.input_folder, file)
    src_cur = []
    texts = read_in_src_texts(file_path)
    texts_all += texts
    
print(len(texts_all))
    
wordfreq = get_wordfreq(texts_all)

json.dump(wordfreq, open(args.output_file, 'w'))