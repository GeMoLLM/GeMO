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
from plot_utils import get_shades, rgb_to_hex, plot_stacked_barchart, plot_grouped_barchart, plot_barchart_any, plot_five_hist

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
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
    d3_src = np.load(f"run_judge_stats/src_stats_final.npz")
    d3_tgt_2 = np.load(f"run_judge_stats/stats_gen_final_codeonly_temp-0.5.npz")
    d3_tgt_4 = np.load(f"run_judge_stats/stats_gen_final_codeonly_temp-0.5_p-1.0.npz")
    d3_tgt_1 = np.load(f"run_judge_stats/stats_gen_final_codeonly.npz")
    d3_tgt_3 = np.load(f"run_judge_stats/stats_gen_final_codeonly_temp-1.0_p-1.0.npz")
else:
    indices = np.load('claude_all_indices.npy')
    d3_src = np.load(f"run_judge_stats/src_stats_final.npz")
    d3_tgt_2 = np.load(f"run_judge_stats/stats_gen_final_claude_codeonly_temp-0.5_p-0.9.npz")
    d3_tgt_4 = np.load(f"run_judge_stats/stats_gen_final_claude_codeonly_temp-0.5_p-1.0.npz")
    d3_tgt_1 = np.load(f"run_judge_stats/stats_gen_final_claude_codeonly_temp-1.0_p-0.9.npz")
    d3_tgt_3 = np.load(f"run_judge_stats/stats_gen_final_claude_codeonly_temp-1.0_p-1.0.npz")

def get_values(data):
    runtime, memories = data['runtimes'], data['memories']
    return runtime.max(axis=2), memories.max(axis=2), \
              runtime.mean(axis=2), memories.mean(axis=2)

d3_src_runtime_max, d3_src_memories_max, d3_src_runtime_mean, d3_src_memories_mean = \
    get_values(d3_src)
if args.claude:
    d3_src_runtime_max = d3_src_runtime_max[indices]
    d3_src_memories_max = d3_src_memories_max[indices]
    d3_src_runtime_mean = d3_src_runtime_mean[indices]
    d3_src_memories_mean = d3_src_memories_mean[indices]

d3_tgt_1_runtime_max, d3_tgt_1_memories_max, d3_tgt_1_runtime_mean, d3_tgt_1_memories_mean = \
    get_values(d3_tgt_1)
d3_tgt_2_runtime_max, d3_tgt_2_memories_max, d3_tgt_2_runtime_mean, d3_tgt_2_memories_mean = \
    get_values(d3_tgt_2)
d3_tgt_3_runtime_max, d3_tgt_3_memories_max, d3_tgt_3_runtime_mean, d3_tgt_3_memories_mean = \
    get_values(d3_tgt_3)
d3_tgt_4_runtime_max, d3_tgt_4_memories_max, d3_tgt_4_runtime_mean, d3_tgt_4_memories_mean = \
    get_values(d3_tgt_4)

plot_five_hist(d3_src_runtime_max.mean(axis=-1), 
                d3_tgt_2_runtime_max.mean(axis=-1), 
                d3_tgt_4_runtime_max.mean(axis=-1), 
                d3_tgt_1_runtime_max.mean(axis=-1),
                d3_tgt_3_runtime_max.mean(axis=-1),
                'fig_code_runtime_max_mean.pdf' if not args.claude else 'fig_code_runtime_max_mean_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='time')

plot_five_hist(d3_src_runtime_mean.mean(axis=-1), 
                d3_tgt_2_runtime_mean.mean(axis=-1), 
                d3_tgt_4_runtime_mean.mean(axis=-1), 
                d3_tgt_1_runtime_mean.mean(axis=-1),
                d3_tgt_3_runtime_mean.mean(axis=-1),
                'fig_code_runtime_mean_mean.pdf' if not args.claude else 'fig_code_runtime_mean_mean_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='time')

plot_five_hist(d3_src_memories_max.mean(axis=-1), 
                d3_tgt_2_memories_max.mean(axis=-1), 
                d3_tgt_4_memories_max.mean(axis=-1), 
                d3_tgt_1_memories_max.mean(axis=-1),
                d3_tgt_3_memories_max.mean(axis=-1),
                'fig_code_memories_max_mean.pdf' if not args.claude else 'fig_code_memories_max_mean_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='space')

plot_five_hist(d3_src_memories_mean.mean(axis=-1), 
                d3_tgt_2_memories_mean.mean(axis=-1), 
                d3_tgt_4_memories_mean.mean(axis=-1), 
                d3_tgt_1_memories_mean.mean(axis=-1),
                d3_tgt_3_memories_mean.mean(axis=-1),
                'fig_code_memories_mean_mean.pdf' if not args.claude else 'fig_code_memories_mean_mean_claude.pdf', 
                colors=modified_palette,
                bins=20,
                mode='space')

plot_five_hist(d3_src_runtime_max.std(axis=-1), 
                d3_tgt_2_runtime_max.std(axis=-1), 
                d3_tgt_4_runtime_max.std(axis=-1), 
                d3_tgt_1_runtime_max.std(axis=-1),
                d3_tgt_3_runtime_max.std(axis=-1),
                'fig_code_runtime_max_std.pdf' if not args.claude else 'fig_code_runtime_max_std_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='time')

plot_five_hist(d3_src_runtime_mean.std(axis=-1), 
                d3_tgt_2_runtime_mean.std(axis=-1), 
                d3_tgt_4_runtime_mean.std(axis=-1), 
                d3_tgt_1_runtime_mean.std(axis=-1),
                d3_tgt_3_runtime_mean.std(axis=-1),
                'fig_code_runtime_mean_std.pdf' if not args.claude else 'fig_code_runtime_mean_std_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='time')

plot_five_hist(d3_src_memories_max.std(axis=-1), 
                d3_tgt_2_memories_max.std(axis=-1), 
                d3_tgt_4_memories_max.std(axis=-1), 
                d3_tgt_1_memories_max.std(axis=-1),
                d3_tgt_3_memories_max.std(axis=-1),
                'fig_code_memories_max_std.pdf' if not args.claude else 'fig_code_memories_max_std_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='space')

plot_five_hist(d3_src_memories_mean.std(axis=-1), 
                d3_tgt_2_memories_mean.std(axis=-1), 
                d3_tgt_4_memories_mean.std(axis=-1), 
                d3_tgt_1_memories_mean.std(axis=-1),
                d3_tgt_3_memories_mean.std(axis=-1),
                'fig_code_memories_mean_std.pdf' if not args.claude else 'fig_code_memories_mean_std_claude.pdf',
                colors=modified_palette,
                bins=20,
                mode='space')
