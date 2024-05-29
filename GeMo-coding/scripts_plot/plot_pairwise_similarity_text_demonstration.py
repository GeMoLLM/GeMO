import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp
import numpy as np
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='gpt-3.5-instruct')
parser.add_argument('--field', '-f', type=str, default='description')
parser.add_argument('--claude', '-c', action='store_true', default=False)
parser.add_argument('--xmax', type=float, default=1.0)
args = parser.parse_args()

TICK_SIZE = 30
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
warnings.filterwarnings("ignore", category=UserWarning)

d3_src = np.load(f"pairwise_sim_scores_{args.field}_{args.model}_src_final.npy")
d3_tgt_2 = np.load(f"pairwise_sim_scores_{args.field}_{args.model}_gen_codeonly_temp-0.5_final.npy")

def plot_kdeplot(data_list, title, colors, out_folder='figs/', alpha=0.1):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['src', 'gen']
    for data, color, label in zip(data_list, colors, labels):
        sns.kdeplot(data=data, fill=True, ax=ax, clip=(0, args.xmax), alpha=alpha, color=color, linewidth=2, label=label)
    plt.xlabel('Similarity', fontsize=32)
    plt.ylabel('Density', fontsize=32)
    plt.xlim(0.35, args.xmax)
    plt.legend(fontsize=28)
    plt.savefig(osp.join(out_folder, title), bbox_inches='tight')

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

# Load the "Paired" palette
palette = sns.color_palette("Paired")
palette_set2 = sns.color_palette("Set2")

# Omit the first color, making the first in the list single, and then pairs follow
modified_palette = [palette_set2[0]] + [palette[7]]

data_list = [d3_src, d3_tgt_2]

plot_kdeplot(
    data_list,
    f'fig_demonstration_code_pairwise_cossim_{args.field}_{args.model}.pdf' if not args.claude else f'fig_code_pairwise_cossim_{args.field}_{args.model}.pdf',
    colors=modified_palette,
    alpha=0.02
)