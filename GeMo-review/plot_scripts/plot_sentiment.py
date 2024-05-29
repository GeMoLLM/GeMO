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

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--metric', type=str, default='mean')
args = parser.parse_args()

folder = 'analysis_results_sentiment'
sent_src = pickle.load(open(osp.join(folder, 'summary_sentiment_goodreads_src_grouped_reviews_long_sub_en_10.pkl'), 'rb'))
sent_gen = pickle.load(
    open(osp.join(folder, f'summary_sentiment_goodreads_{args.mode}_{args.model}-chat_500.pkl'), 'rb'))

books = list(sent_src.keys())

sent_x_src = []
sent_x_gen = []
for book in books:
    sent_x_src.append(sent_src[book][args.metric])
    sent_x_gen.append(sent_gen[book][args.metric])

sent_x_src = np.array(sent_x_src)
sent_x_gen = np.array(sent_x_gen)

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

out_folder = 'figs/'
os.makedirs(out_folder, exist_ok=True)

def plot_ridge_plot(str_list, data_list, filename):
    data = {
        s: d for s, d in zip(str_list, data_list)
    }
    data_combined = pd.DataFrame(data).melt(var_name='List', value_name='Value')

    # kde_max_values = -1
    # for name, group in data_combined.groupby('List'):
    #     kde = gaussian_kde(group['Value'])
    #     kde_max_values = max(kde_max_values, np.max(kde(group['Value'])))

    # Creating the ridge plot
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    g = sns.FacetGrid(data_combined, row="List", hue="List", aspect=5, height=1, palette="Set2")

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "Value", clip_on=(0,1), fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "Value", clip_on=(0,1), color="w", linewidth=2)
    g.map(plt.axhline, y=0, lw=0.2, clip_on=(0,1))

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)
        ax.set_ylabel('')
        # ax.set_xticklabels(fontsize=16)

    g.map(label, "Value")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.1)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.xlabel('')
    plt.ylabel('')

    # plt.xlim(0, 1)
    plt.xticks(fontsize=16)
    plt.show()
    plt.savefig(
        osp.join(out_folder, filename), bbox_inches='tight')
    
for ti, T in enumerate(T_list):
    str_list = ['src'] + [f'T={T:.1f}, p={P:.2f}' for P in P_list]
    data_list = [sent_x_src] + [sent_x_gen[:,ti,i] for i in range(4)]
    plot_ridge_plot(str_list, data_list, f'sentiment_{args.metric}_{args.mode}_{args.model}_T-{T}.pdf')
    print(f'done plotting for T = {T}!')
    
for pi, P in enumerate(P_list):
    str_list = ['src'] + [f'T={T:.1f}, p={P:.2f}' for T in T_list]
    data_list = [sent_x_src] + [sent_x_gen[:,i,pi] for i in range(4)]
    plot_ridge_plot(str_list, data_list, f'sentiment_{args.metric}_{args.mode}_{args.model}_P-{P}.pdf')
    print(f'done plotting for P = {P}!')
    