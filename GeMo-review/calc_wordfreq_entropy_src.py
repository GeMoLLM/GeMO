import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyze_utils import calculate_counter_entropy

folder = 'analysis_results_wordfreq'
src_path = osp.join(folder, 'summary_wordfreq_goodreads_src_grouped_reviews_long_sub_en_10.pkl')

src_data = pickle.load(open(src_path, 'rb'))

src_counter = src_data
entropy = calculate_counter_entropy(src_counter)

out_folder = 'tables'
os.makedirs(out_folder, exist_ok=True)
np.save(f'{out_folder}/wordfreq_entropy_src-chat_500.npy', entropy)
print(f'entropy = {entropy:.2f}')