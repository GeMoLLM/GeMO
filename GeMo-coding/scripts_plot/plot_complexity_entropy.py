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
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--claude', '-c', action='store_true', default=False)
args = parser.parse_args()

TICK_SIZE = 30
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")
modified_palette = [palette_set2[0]] + palette[6:10]

if not args.claude:
    d3_src = np.load(f"complexity_train_solution_entropy_{args.model}_final.npz")
    d3_tgt_2 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_codeonly_temp-0.5.npz")
    d3_tgt_4 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_codeonly_temp-0.5_p-1.0.npz")
    d3_tgt_1 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_codeonly.npz")
    d3_tgt_3 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_codeonly_temp-1.0_p-1.0.npz")
else:
    indices = np.load('claude_all_indices.npy')
    d3_src = np.load(f"complexity_train_solution_entropy_{args.model}_final.npz")
    d3_tgt_2 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_claude_codeonly_temp-0.5_p-0.9.npz")
    d3_tgt_4 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_claude_codeonly_temp-0.5_p-1.0.npz")
    d3_tgt_1 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_claude_codeonly_temp-1.0_p-0.9.npz")
    d3_tgt_3 = np.load(f"complexity_train_solution_entropy_gen_final_{args.model}_claude_codeonly_temp-1.0_p-1.0.npz")
    
print(d3_src)
N = 100

te_src, se_src = d3_src['time_complexity_entropy'], d3_src['space_complexity_entropy']
if args.claude:
    te_src, se_src = te_src[indices], se_src[indices]
te_tgt_1, se_tgt_1 = d3_tgt_1['time_complexity_entropy'], d3_tgt_1['space_complexity_entropy']
te_tgt_2, se_tgt_2 = d3_tgt_2['time_complexity_entropy'], d3_tgt_2['space_complexity_entropy']
te_tgt_3, se_tgt_3 = d3_tgt_3['time_complexity_entropy'], d3_tgt_3['space_complexity_entropy']
te_tgt_4, se_tgt_4 = d3_tgt_4['time_complexity_entropy'], d3_tgt_4['space_complexity_entropy']

plot_barchart_any(
    [te_src, te_tgt_2, te_tgt_4, te_tgt_1, te_tgt_3], 
    f'fig_code_time_complexity_entropy_{args.model}.pdf' if not args.claude else f'fig_code_time_complexity_entropy_{args.model}_claude.pdf',
    xlow=-0.5, xhigh=5.5,
    group=0.4,
    width=0.8,
    palette=modified_palette)

plot_barchart_any(
    [se_src, se_tgt_2, se_tgt_4, se_tgt_1, se_tgt_3], 
    f'fig_code_space_complexity_entropy_{args.model}.pdf' if not args.claude else f'fig_code_space_complexity_entropy_{args.model}_claude.pdf',
    xlow=-0.5, xhigh=3.5,
    group=0.4,
    width=0.8,
    palette=modified_palette,
    figsize=(10,3))