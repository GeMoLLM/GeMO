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
from plot_utils import get_shades, rgb_to_hex, read_data, \
    plot_stacked_barchart, plot_custom_barcharts, plot_kdeplot, read_distr_data, plot_grouped_barchart, \
    get_count_indexed_topics
import umap.umap_ as UMAP
from bertopic import BERTopic

TICK_SIZE = 16
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='personalized')
args = parser.parse_args()

folder = 'analysis_results_topic_distr'
src_path = osp.join(folder, 'summary_topic_distr_goodreads_src_grouped_reviews_long_sub_en_10.pkl')
gen_path_1 = osp.join(folder, f'summary_topic_distr_goodreads_personalized_gpt-3.5-instruct-chat_500.pkl')
gen_path_2 = osp.join(folder, f'summary_topic_distr_goodreads_personation_gpt-3.5-instruct-chat_500.pkl')
gen_path_3 = osp.join(folder, f'summary_topic_distr_goodreads_personalized_gpt-4-chat_500.pkl')
gen_path_4 = osp.join(folder, f'summary_topic_distr_goodreads_personation_gpt-4-chat_500.pkl')

uni_src, cnt_src = read_distr_data(src_path)
N_src = sum(cnt_src)
data_gen_1 = read_distr_data(gen_path_1)
data_gen_2 = read_distr_data(gen_path_2)
data_gen_3 = read_distr_data(gen_path_3)
data_gen_4 = read_distr_data(gen_path_4)

indices_src = np.argsort(cnt_src)[::-1][:10]

uni_src_sel = uni_src[indices_src]
cnt_src_sel = cnt_src[indices_src]

str_list = []
for topic in uni_src_sel:
    topic_str = topic_model.get_topic(topic)
    str_list.append(str(topic)+'_'+topic_str[0][0]+'_'+topic_str[1][0])

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

# Load the "Paired" palette
palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")

# Omit the first color, making the first in the list single, and then pairs follow
modified_palette = [palette_set2[0]] + palette[0:4]

data_list = [list(cnt_src_sel/N_src)]

T = 1.2
P = 1.0
uni_gen, cnt_gen = data_gen_1[T][P]
cnt_gen_sel = get_count_indexed_topics(uni_src_sel, uni_gen, cnt_gen)
data_list.append([x/sum(cnt_gen) for x in cnt_gen_sel])

uni_gen, cnt_gen = data_gen_2[T][P]
cnt_gen_sel = get_count_indexed_topics(uni_src_sel, uni_gen, cnt_gen)
data_list.append([x/sum(cnt_gen) for x in cnt_gen_sel])

uni_gen, cnt_gen = data_gen_3[T][P]
cnt_gen_sel = get_count_indexed_topics(uni_src_sel, uni_gen, cnt_gen)
data_list.append([x/sum(cnt_gen) for x in cnt_gen_sel])

uni_gen, cnt_gen = data_gen_4[T][P]
cnt_gen_sel = get_count_indexed_topics(uni_src_sel, uni_gen, cnt_gen)
data_list.append([x/sum(cnt_gen) for x in cnt_gen_sel])

plot_grouped_barchart(
    str_list,
    data_list,
    f'fig_topic_distr_{args.mode}_gpt_cmp.pdf',
    figsize=(9,8),
    xmax=0.15,
    palette=modified_palette,
    noyticks=False
)
