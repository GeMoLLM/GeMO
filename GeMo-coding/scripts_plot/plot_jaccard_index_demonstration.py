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
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--claude', '-c', action='store_true', default=False)
parser.add_argument('--field', '-f', type=str, default='tags')
args = parser.parse_args()

TICK_SIZE = 20
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=26)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

d3_src = np.load(f"{args.field}_train_solution_similarity_{args.model}_final.npy")
d3_tgt_2 = np.load(f"{args.field}_train_solution_similarity_gen_codeonly_temp-0.5_{args.model}_final.npy")
    
out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

bins = np.linspace(0, 1, 6)
N = 100
bar_width = 0.8

palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")
modified_palette = [palette_set2[0]] + [palette[7]]
input_color = [rgb_to_hex(color) for color in modified_palette]
shades = [get_shades(color)[::-1] for color in input_color]
shades = np.array(shades).T[1:]
print(shades.shape)
shades = np.array([['#444444', '#666666', '#888888', '#AAAAAA', '#CCCCCC'],
          ['#444444', '#666666', '#888888', '#AAAAAA', '#CCCCCC']]).T

str_list = [
    'src', 
    'gen',
]

data_list = [
    d3_src,
    d3_tgt_2,
]

plot_stacked_barchart(
    str_list, data_list,
    f'fig_demonstration_{args.field}_train_solution_similarity_stacked_barchart_{args.model}.pdf' if not args.claude else f'fig_{args.field}_train_solution_similarity_stacked_barchart_{args.model}.pdf',
    N=N,
    bins=bins,
    shades=shades,
)