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
args = parser.parse_args()

data1 = np.load(f'tables/wordfreq_cossim_personalized_{args.model}-chat_500.npy')
data2 = np.load(f'tables/wordfreq_cossim_personation_{args.model}-chat_500.npy')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

data = [data1[1], data2[1], data1[3], data2[3]]
# Plot the similarity matrix as a heatmap with increased precision
sns.heatmap(data, annot=True, cmap="YlGnBu", ax=ax, annot_kws={"fontsize":18}, fmt=".3f", vmin=0.269, vmax=0.366)

# Adjust the tick labels and rotation
ax.set_xticks(np.arange(4) + 0.5)
ax.set_yticks(np.arange(4) + 0.5)
ax.set_xticklabels(['$p=0.9$', '$p=0.95$', '$p=0.98$', '$p=1.0$'], fontsize=16)
ax.set_yticklabels(['$T=0.5$', '$T=0.8$', '$T=1.0$', '$T=1.2$'], fontsize=16)

# Add a colorbar
cbar = ax.collections[0].colorbar

# Increase the fontsize of the colorbar ticks
cbar.ax.tick_params(labelsize=16)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.savefig(f'figs/wordfreq_cossim_{args.mode}_{args.model}-chat_500.pdf', bbox_inches='tight')