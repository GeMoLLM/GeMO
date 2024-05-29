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
from plot_utils import get_shades, rgb_to_hex, read_data, \
    plot_stacked_barchart, plot_custom_barcharts, plot_kdeplot

warnings.filterwarnings("ignore", category=UserWarning)

TICK_SIZE = 30
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--metric', type=str, default='entropy')
args = parser.parse_args()

folder = 'analysis_results_topic'
src_path = osp.join(folder, 'summary_topic_goodreads_src_grouped_reviews_long_sub_en_10.pkl')
gen_path = osp.join(folder, f'summary_topic_goodreads_{args.mode}_{args.model}-chat_500.pkl')

sent_src = pickle.load(open(src_path, 'rb'))

books = list(sent_src.keys())

sent_x_src = read_data(src_path, books, args.metric)
sent_x_gen = read_data(gen_path, books, args.metric)

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

for ti, T in enumerate(T_list):
    data_list = [sent_x_src] + [sent_x_gen[:,ti,i] for i in range(4)]
    plot_kdeplot(
        data_list, 
        f'topic_{args.metric}_{args.mode}_{args.model}_T-{T}.pdf',
        colors=sns.color_palette("Set2"),
        alpha=0.02
    )
    # plot_custom_barcharts(
    #     data_list, 
    #     f'topic_{args.metric}_{args.mode}_{args.model}_T-{T}.pdf',
    #     -0.5, 5.5, width=0.8, precision=2, xrange=0.5, ymax=0.45,
    #     palette=sns.color_palette("Set2"))
    print(f'done plotting for T = {T}!')
    
modified_palette = [sns.color_palette("Set2")[0]] + sns.color_palette("tab10")[1:5]
for pi, P in enumerate(P_list):
    data_list = [sent_x_src] + [sent_x_gen[:,i,pi] for i in range(4)]
    plot_kdeplot(
        data_list, 
        f'topic_{args.metric}_{args.mode}_{args.model}_P-{P}.pdf',
        colors=modified_palette)
    # plot_custom_barcharts(
    #     data_list, 
    #     f'topic_{args.metric}_{args.mode}_{args.model}_P-{P}.pdf',
    #     -0.5, 5.5, width=0.8, precision=2, xrange=0.5, ymax=0.45,
    #     palette=modified_palette)
    print(f'done plotting for P = {P}!')
    