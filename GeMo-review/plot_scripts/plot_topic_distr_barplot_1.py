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
    plot_stacked_barchart, plot_custom_barcharts, plot_kdeplot, read_distr_data, plot_grouped_barchart, \
    get_count_indexed_topics
import umap.umap_ as UMAP
from bertopic import BERTopic

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")

TICK_SIZE = 16
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
args = parser.parse_args()

folder = 'analysis_results_topic_distr'
src_path = osp.join(folder, 'summary_topic_distr_goodreads_src_grouped_reviews_long_sub_en_10.pkl')
gen_path = osp.join(folder, f'summary_topic_distr_goodreads_{args.mode}_{args.model}-chat_500.pkl')

uni_src, cnt_src = read_distr_data(src_path)
N_src = sum(cnt_src)
data_gen = read_distr_data(gen_path)

indices_src = np.argsort(cnt_src)[::-1][:10]

uni_src_sel = uni_src[indices_src]
cnt_src_sel = cnt_src[indices_src]

str_list = []
for topic in uni_src_sel:
    topic_str = topic_model.get_topic(topic)
    str_list.append(str(topic)+':{'+topic_str[0][0]+','+topic_str[1][0]+'..}')

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

for ti, T in enumerate(T_list):
    if T != 1.2:
        continue
    data_list = [list(cnt_src_sel/N_src)]
    for P in P_list:
        uni_gen, cnt_gen = data_gen[T][P]
        cnt_gen_sel = get_count_indexed_topics(uni_src_sel, uni_gen, cnt_gen)
        data_list.append([x/sum(cnt_gen) for x in cnt_gen_sel])
        
    plot_grouped_barchart(
        str_list,
        data_list,
        f'fig_topic_distr_{args.mode}_{args.model}_T-{T}_single.pdf',
        group_list=['src', '$T=1.2,p=0.9$', '$T=1.2,p=0.95$', '$T=1.2,p=0.98$', '$T=1.2,p=1.0$'],
        figsize=(8,8),
        noyticks=False,
        xmax=0.2,
        legend=True,
        palette=sns.color_palette("Set2")
    )
    print(f'done plotting for T = {T}!')