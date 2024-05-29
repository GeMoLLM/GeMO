import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--front', action='store_true', default=False)
args = parser.parse_args()

data = np.load(f'tables/wordfreq_entropy_{args.mode}_{args.model}-chat_500.npy')

if args.front:
    figsize = (5,4)
else:
    figsize = (6,4)
# Create a figure and axis
fig, ax = plt.subplots(figsize=figsize)

# Plot the similarity matrix as a heatmap with increased precision
sns.heatmap(data, annot=True, cmap="YlGnBu", 
            ax=ax, annot_kws={"fontsize":18}, fmt=".2f", 
            vmin=8.90, vmax=9.75, cbar=not args.front)

# Adjust the tick labels and rotation
ax.set_xticks(np.arange(4) + 0.5)
ax.set_yticks(np.arange(4) + 0.5)
ax.set_xticklabels(['$p=0.9$', '$p=0.95$', '$p=0.98$', '$p=1.0$'], fontsize=16)
ax.set_yticklabels(['$T=0.5$', '$T=0.8$', '$T=1.0$', '$T=1.2$'], fontsize=16)

if not args.front:
    # Add a colorbar
    cbar = ax.collections[0].colorbar

    # Increase the fontsize of the colorbar ticks
    cbar.ax.tick_params(labelsize=16)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.savefig(f'figs/wordfreq_entropy_{args.mode}_{args.model}-chat_500.pdf', bbox_inches='tight')