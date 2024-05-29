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
from plot_utils import get_shades, rgb_to_hex, plot_stacked_barchart, plot_grouped_barchart, plot_barchart_any

parser = argparse.ArgumentParser()
parser.add_argument('--claude', '-c', action='store_true', default=False)
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--N', type=int, default=100)
args = parser.parse_args()

TICK_SIZE = 30
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")
modified_palette = [palette_set2[0]] + palette[6:10]

if not args.claude:
    d3_src = np.load(f"copydetect_scores_src_final/all_scores.npy")
    d3_tgt_2 = np.load(f"copydetect_scores_gen_final_gpt4_codeonly_temp-0.5/all_scores.npy")
    d3_tgt_4 = np.load(f"copydetect_scores_gen_final_gpt4_codeonly_temp-0.5_p-1.0/all_scores.npy")
    d3_tgt_1 = np.load(f"copydetect_scores_gen_final_gpt4/all_scores.npy")
    d3_tgt_3 = np.load(f"copydetect_scores_gen_final_gpt4_codeonly_temp-1.0_p-1.0/all_scores.npy")
else:
    indices = np.load('claude_all_indices.npy')
    d3_src = np.load(f"copydetect_scores_src_final/all_scores.npy")[indices]
    d3_tgt_2 = np.load(f"copydetect_scores_gen_final_claude_claude_codeonly_temp-0.5_p-0.9/all_scores.npy")
    d3_tgt_4 = np.load(f"copydetect_scores_gen_final_claude_claude_codeonly_temp-0.5_p-1.0/all_scores.npy")
    d3_tgt_1 = np.load(f"copydetect_scores_gen_final_claude_claude_codeonly_temp-1.0_p-0.9/all_scores.npy")
    d3_tgt_3 = np.load(f"copydetect_scores_gen_final_claude_claude_codeonly_temp-1.0_p-1.0/all_scores.npy")
    
N = args.N

# plot_barchart_any(
#     [d3_src.mean(axis=1), d3_tgt_2.mean(axis=1), d3_tgt_1.mean(axis=1)], 
#     f'fig_code_plagiarism_score.pdf',
#     xlow=-1, xhigh=12,
#     group=0.1,
#     width=1.2,
#     precision=1,
#     palette=modified_palette)

all_data = np.concatenate([d3_src, d3_tgt_1, d3_tgt_2])
bins = np.linspace(all_data.min(), all_data.max(), num=20)  # 50 bins, or choose a number that works well for your data

sns.histplot(d3_src.mean(axis=1), element='step', bins=bins, stat='probability', alpha=0.3, linewidth=0.5, color=modified_palette[0])
sns.histplot(d3_tgt_2.mean(axis=1), kde=True, element='step', bins=bins, stat='probability', alpha=0.3, linewidth=0.5, color=modified_palette[1])
sns.histplot(d3_tgt_4.mean(axis=1), kde=True, element='step', bins=bins, stat='probability', alpha=0.3, linewidth=0.5, color=modified_palette[2])
sns.histplot(d3_tgt_1.mean(axis=1), kde=True, element='step', bins=bins, stat='probability', alpha=0.3, linewidth=0.5, color=modified_palette[3])
sns.histplot(d3_tgt_3.mean(axis=1), kde=True, element='step', bins=bins, stat='probability', alpha=0.3, linewidth=0.5, color=modified_palette[4])
plt.ylabel('')

outfile = osp.join('figs', 'fig_code_plagiarism_score_hist.pdf') \
    if not args.claude else osp.join('figs', 'fig_code_plagiarism_score_hist_claude.pdf')
plt.savefig(outfile, bbox_inches='tight')