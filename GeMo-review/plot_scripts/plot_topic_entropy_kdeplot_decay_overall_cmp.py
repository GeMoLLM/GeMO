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
from plot_utils import get_shades, rgb_to_hex, read_data, plot_custom_barcharts, plot_kdeplot

TICK_SIZE = 30
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--metric', type=str, default='entropy')
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

folder = 'analysis_results_topic'
src_path = osp.join(folder, 'summary_topic_goodreads_src_grouped_reviews_long_sub_en_10.pkl')
gen_path_1 = osp.join(folder, f'summary_topic_goodreads_personalized_{args.model}-chat_500.pkl')
gen_path_2 = osp.join(folder, f'summary_topic_goodreads_personation_{args.model}-chat_500.pkl')
gen_path_decay_1 = osp.join(folder, f'summary_topic_goodreads_decay_personalized_{args.model}-chat_500.pkl') \
    if args.end_temperature == 1.0 and args.period == 20 \
        else osp.join(folder, f'summary_topic_goodreads_decay-{args.end_temperature}-{args.period}_personalized_{args.model}-chat_500.pkl')
gen_path_decay_2 = osp.join(folder, f'summary_topic_goodreads_decay_personation_{args.model}-chat_500.pkl') \
    if args.end_temperature == 1.0 and args.period == 20 \
        else osp.join(folder, f'summary_topic_goodreads_decay-{args.end_temperature}-{args.period}_personation_{args.model}-chat_500.pkl')
        
sent_src = pickle.load(open(src_path, 'rb'))

books = list(sent_src.keys())

sent_x_src = read_data(src_path, books, args.metric)
sent_x_gen_1 = read_data(gen_path_1, books, args.metric)
sent_x_gen_2 = read_data(gen_path_2, books, args.metric)
sent_x_gen_decay_1 = read_data(gen_path_decay_1, books, args.metric)
sent_x_gen_decay_2 = read_data(gen_path_decay_2, books, args.metric)

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

# Load the "Paired" palette
palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")

# Omit the first color, making the first in the list single, and then pairs follow
modified_palette = [palette_set2[0]] + palette[6:10]

data_list = [sent_x_src,
             sent_x_gen_1[:,3,3],
             sent_x_gen_decay_1[:,0,3],
             sent_x_gen_2[:,3,3],
             sent_x_gen_decay_2[:,0,3],
             ]

out_file = f'fig_topic_histplot_{args.metric}_{args.model}_decay_mode_cmp.pdf' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'fig_topic_histplot_{args.metric}_{args.model}_decay-{args.end_temperature}-{args.period}_mode_cmp.pdf'
plot_kdeplot(
    data_list,
    out_file,
    colors=modified_palette,
    alpha=0.02
)