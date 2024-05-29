import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
import os.path as osp
import os
import warnings
from scipy.stats import gaussian_kde
import colorsys

TICK_SIZE = 20
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--metric', type=str, default='mean')
args = parser.parse_args()

folder = 'analysis_results_sentiment'
sent_src = pickle.load(open(osp.join(folder, 'summary_sentiment_goodreads_src_grouped_reviews_long_sub_en_10.pkl'), 'rb'))
sent_gen = pickle.load(
    open(osp.join(folder, f'summary_sentiment_goodreads_{args.mode}_{args.model}-chat_500.pkl'), 'rb'))

books = list(sent_src.keys())

sent_x_src = []
sent_x_gen = []
for book in books:
    sent_x_src.append(sent_src[book][args.metric])
    sent_x_gen.append(sent_gen[book][args.metric])

sent_x_src = np.array(sent_x_src)
sent_x_gen = np.array(sent_x_gen)

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

bins = [-0.05, 0.35, 0.55, 0.75, 0.85, 0.95, 1.05]
N = 742
bar_width = 0.8

def calc_distance(x1, x2):
    return np.mean(np.abs(x1 - x2))

dist_list = np.zeros((len(T_list), len(P_list)))
for ti, T in enumerate(T_list):
    for pi, P in enumerate(P_list):
        assert -1 not in sent_x_gen[:,ti,pi]
        dist_list[ti][pi] = calc_distance(sent_x_src, sent_x_gen[:,ti,pi])

out_folder = 'tables'
os.makedirs(out_folder, exist_ok=True)
out_file = f'{out_folder}/sentiment_distance_{args.model}_{args.mode}.csv'
# save dist_list to out_file
np.savetxt(out_file, dist_list, delimiter=',')