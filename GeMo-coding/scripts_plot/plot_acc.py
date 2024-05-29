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
from plot_utils import get_shades, rgb_to_hex, plot_stacked_barchart

parser = argparse.ArgumentParser()
parser.add_argument('--claude', '-c', action='store_true', default=False)
args = parser.parse_args()

TICK_SIZE = 20
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

if not args.claude:
    d3_src = np.load(f"acc_train_src.npy")
    d3_tgt_2 = np.load(f"acc_train_gpt4_codeonly_temp-0.5.npy")
    d3_tgt_4 = np.load(f"acc_train_gpt4_codeonly_temp-0.5_p-1.0.npy")
    d3_tgt_1 = np.load(f"acc_train_gpt4_codeonly.npy")
    d3_tgt_3 = np.load(f"acc_train_gpt4_codeonly_temp-1.0_p-1.0.npy")
else:
    indices = np.load('claude_all_indices.npy')
    d3_src = np.load(f"acc_train_src.npy")[indices]
    d3_tgt_2 = np.load(f"acc_train_claude_codeonly_temp-0.5_p-0.9.npy")
    d3_tgt_4 = np.load(f"acc_train_claude_codeonly_temp-0.5_p-1.0.npy")
    d3_tgt_1 = np.load(f"acc_train_claude_codeonly_temp-1.0_p-0.9.npy")
    d3_tgt_3 = np.load(f"acc_train_claude_codeonly_temp-1.0_p-1.0.npy")

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

bins = np.linspace(0, 1, 6)
N = 100
bar_width = 0.8

palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")
modified_palette = [palette_set2[0]] + palette[6:10]
input_color = [rgb_to_hex(color) for color in modified_palette]
shades = [get_shades(color)[::-1] for color in input_color]
shades = np.array(shades).T[1:]
print(shades.shape)

str_list = [
    'src', 
    '$T=0.5$\n$p=0.9$',
    '$T=0.5$\n$p=1.0$',
    '$T=1.0$\n$p=0.9$',
    '$T=1.0$\n$p=1.0$'
]

data_list = [
    d3_src,
    d3_tgt_2,
    d3_tgt_4,
    d3_tgt_1,
    d3_tgt_3,
]

plot_stacked_barchart(
    str_list, data_list,
    f'fig_code_acc_stacked_barchart.pdf' if not args.claude else f'fig_code_acc_stacked_barchart_claude.pdf',
    N=N,
    bins=bins,
    shades=shades,
)