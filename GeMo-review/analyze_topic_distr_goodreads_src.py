from transformers import pipeline
import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import json
import pickle
from metrics import get_mean, get_std, get_entropy, transform_sentiment

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--out_file', type=str, required=True)
args = parser.parse_args()

result = []

data = json.load(open(args.input_file))
topics = []
for k, v in data.items():
    topics += v
result = np.unique(topics, return_counts=True)

pickle.dump(result, open(args.out_file, 'wb'))
