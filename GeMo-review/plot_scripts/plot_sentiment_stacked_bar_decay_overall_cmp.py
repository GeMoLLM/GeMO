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
from plot_utils import get_shades, rgb_to_hex, read_data, plot_stacked_barchart

TICK_SIZE = 20
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--metric', type=str, default='mean')
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

folder = 'analysis_results_sentiment'
src_path = osp.join(folder, 'summary_sentiment_goodreads_src_grouped_reviews_long_sub_en_10.pkl')
gen_path_1 = osp.join(folder, f'summary_sentiment_goodreads_personalized_{args.model}-chat_500.pkl')
gen_path_2 = osp.join(folder, f'summary_sentiment_goodreads_personation_{args.model}-chat_500.pkl')
gen_path_decay_1 = osp.join(folder, f'summary_sentiment_goodreads_decay_personalized_{args.model}-chat_500.pkl') \
    if args.end_temperature == 1.0 and args.period == 20 \
        else osp.join(folder, f'summary_sentiment_goodreads_decay_personalized-{args.end_temperature}-{args.period}_{args.model}-chat_500.pkl')
gen_path_decay_2 = osp.join(folder, f'summary_sentiment_goodreads_decay_personation_{args.model}-chat_500.pkl') \
    if args.end_temperature == 1.0 and args.period == 20 \
        else osp.join(folder, f'summary_sentiment_goodreads_decay_personation-{args.end_temperature}-{args.period}_{args.model}-chat_500.pkl')

sent_src = pickle.load(open(src_path, 'rb'))
books = list(sent_src.keys())

sent_x_src = read_data(src_path, books, args.metric)
sent_x_gen_1 = read_data(gen_path_1, books, args.metric)
sent_x_gen_decay_1 = read_data(gen_path_decay_1, books, args.metric)
sent_x_gen_2 = read_data(gen_path_2, books, args.metric)
sent_x_gen_decay_2 = read_data(gen_path_decay_2, books, args.metric)

# Load the "Paired" palette
palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")

# Omit the first color, making the first in the list single, and then pairs follow
modified_palette = [palette_set2[0]] + palette[6:10]
input_color = [rgb_to_hex(color) for color in modified_palette]
shades = [get_shades(color)[::-1] for color in input_color]
shades = np.array(shades).T
print(shades.shape)

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

str_list = ['src', 
            '$T=1.0$\n(1)', 'lin\n(1)',
            '$T=1.0$\n(2)', 'lin\n(2)',]
data_list = [
    sent_x_src,
    sent_x_gen_1[:,2,3],
    sent_x_gen_decay_1[:,0,3],
    sent_x_gen_2[:,2,3],
    sent_x_gen_decay_2[:,0,3],
]

out_file = f'fig_sentiment_decay_stacked_barchart_{args.model}_cmp.pdf' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'fig_sentiment_decay_stacked_barchart_{args.model}-{args.end_temperature}-{args.period}_cmp.pdf'
plot_stacked_barchart(str_list, data_list, out_file, shades=shades)
