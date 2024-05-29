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

folder = 'analysis_results_wordfreq'
src_path = osp.join(folder, 'summary_wordfreq_goodreads_src_grouped_reviews_long_sub_en_10.pkl')

src_data = pickle.load(open(src_path, 'rb'))

src_counter = src_data
count = len(src_counter)

out_folder = 'tables'
os.makedirs(out_folder, exist_ok=True)
np.save(f'{out_folder}/wordfreq_count_src-chat_500.npy', count)
print(f'count = {count}')