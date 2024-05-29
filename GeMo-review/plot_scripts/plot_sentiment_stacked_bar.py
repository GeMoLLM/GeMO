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

TICK_SIZE = 20
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=TICK_SIZE)
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

bins = [-0.05, 0.35, 0.55, 0.75, 0.85, 0.95, 1.05]
N = 742
bar_width = 0.8

def get_shades(color_hex, n=6):
    """
    Returns a list of n shades of the given color, from darkest to lightest.
    
    Parameters:
    color_hex (str): The color code in hexadecimal format (e.g., "#FF0000" for red).
    n (int, optional): The number of shades to generate. Default is 6.
    
    Returns:
    list: A list of n hexadecimal color codes representing the shades of the input color.
    """
    # Convert the hexadecimal color code to RGB values
    r, g, b = [int(color_hex[i:i+2], 16) for i in (1, 3, 5)]
    
    # Convert RGB values to HSV (Hue, Saturation, Value)
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    
    # Create a list of value (brightness) levels
    values = [v * (1 - (i / (n - 1))) for i in range(n)]
    
    # Generate the shades by converting HSV back to RGB and then to hexadecimal
    shades = []
    for value in values:
        rgb = colorsys.hsv_to_rgb(h, s, value)
        r, g, b = [round(255 * c) for c in rgb]
        shades.append(f"#{r:02X}{g:02X}{b:02X}")
    
    return shades

# Example usage
input_color = ["#66c2a5", '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
shades = [get_shades(color)[::-1] for color in input_color]
shades = np.array(shades).T

def plot_stacked_barchart(str_list, data_list, filename):
    hist_list = []
    for data in data_list:
        hist, _ = np.histogram(data, bins)
        hist_list.append(hist)

    fig, ax = plt.subplots()

    # Positions of the bars on the x-axis
    bar_positions = np.arange(5)

    # Stacking the bars for each range
    for i in range(len(bins) - 1):
        # Heights of the bars for this range
        bar_heights = [hist[i]/N for hist in hist_list]
        # The bottom position for the bars
        if i == 0:
            bottoms = np.zeros(5)
        else:
            bottoms += bar_heights_previous
        # Plot the bars
        ax.bar(bar_positions, bar_heights, bar_width, bottom=bottoms, label=f'({bins[i]:.1f},{bins[i+1]:.1f})', color=shades[i])
        bar_heights_previous = bar_heights

    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # ax.set_xlabel('List')
    # ax.set_ylabel('Counts', fontsize=24)
    # ax.set_title('Stacked Bar Plot of Data with Lists on x-axis')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(str_list, fontsize=15)
    # ax.legend(loc='best')
    plt.savefig(osp.join(out_folder, filename), bbox_inches='tight')
    
for ti, T in enumerate(T_list):
    str_list = ['src'] + ["$T=" + str(T) + "$\n$p=" + str(P)+"$" for P in P_list]
    data_list = [sent_x_src] + [sent_x_gen[:,ti,i] for i in range(4)]
    plot_stacked_barchart(str_list, data_list, f'fig_sentiment_stacked_barchart_{args.metric}_{args.mode}_{args.model}_T-{T}.pdf')
    print(f'done plotting for T = {T}!')
    
for pi, P in enumerate(P_list):
    str_list = ['src'] + ["$T=" + str(T) + "$\n$p=" + str(P)+"$\n" for T in T_list]
    data_list = [sent_x_src] + [sent_x_gen[:,i,pi] for i in range(4)]
    plot_stacked_barchart(str_list, data_list, f'fig_sentiment_stacked_barchart_{args.metric}_{args.mode}_{args.model}_P-{P}.pdf')
    print(f'done plotting for P = {P}!')
