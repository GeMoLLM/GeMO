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
from plot_utils import get_shades, rgb_to_hex, plot_stacked_barchart, plot_grouped_barchart

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--claude', '-c', action='store_true', default=False)
args = parser.parse_args()

TICK_SIZE = 20
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

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
time_all_src, space_all_src = d3_src['time_all'], d3_src['space_all']
if args.claude:
    time_all_src = time_all_src[indices]
    space_all_src = space_all_src[indices]
time_all_tgt_1, space_all_tgt_1 = d3_tgt_1['time_all'], d3_tgt_1['space_all']
time_all_tgt_2, space_all_tgt_2 = d3_tgt_2['time_all'], d3_tgt_2['space_all']
time_all_tgt_4, space_all_tgt_4 = d3_tgt_4['time_all'], d3_tgt_4['space_all']
time_all_tgt_3, space_all_tgt_3 = d3_tgt_3['time_all'], d3_tgt_3['space_all']

str_rev_map = {
    1: 'O($1$)',
    2: 'O($\log n$)',
    3: 'O($\sqrt{n}$)',
    4: 'O($n$)',
    5: 'O($n \log n$)',
    6: 'O($n \sqrt{n}$)',
    7: 'O($n^2$)',
    8: 'O($n^2 \log n$)',
    9: 'O($n^3$)',
    10: 'O($n^4$)',
    11: 'O($2^n$)',
}

def get_count(arr):
    unique, count = np.unique(arr.flatten(), return_counts=True)
    count_map = dict(zip(unique, count))
    ans = np.zeros(11, dtype=int)
    for i in range(11):
        ans[i] = count_map.get(i+1, 0)
    return ans
    
palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")
modified_palette = [palette_set2[0]] + palette[6:10]

def produce_plot(src, tgt_1, tgt_2, tgt_3, tgt_4, N, mode):
    indices = [0,1,2,3,4,6] if mode == 'time' else [0,3,6]

    data_list = [
        get_count(src) / N,
        get_count(tgt_2) / N,
        get_count(tgt_4) / N,
        get_count(tgt_1) / N,
        get_count(tgt_3) / N,
    ]
    
    data_list = [l[indices] for l in data_list]
    
    print(data_list)
    str_list = list(str_rev_map.values())
    
    plot_grouped_barchart(
        [str_list[x] for x in indices],
        data_list,
        f'fig_code_{mode}_complexity_{args.model}.pdf' if not args.claude else f'fig_code_{mode}_complexity_{args.model}_claude.pdf',
        figsize=(9,6) if mode == 'time' else (9,3),
        noyticks=False,
        palette=modified_palette,
        xmax=0.6,
    )

N = time_all_src.flatten().shape[0]
produce_plot(time_all_src, time_all_tgt_1, time_all_tgt_2, time_all_tgt_3, time_all_tgt_4, N, 'time')
produce_plot(space_all_src, space_all_tgt_1, space_all_tgt_2, space_all_tgt_3, space_all_tgt_4, N, 'space')