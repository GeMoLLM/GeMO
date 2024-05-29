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
import json

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


def rgb_to_hex(rgb_tuple):
    """
    Converts an RGB tuple (values between 0 and 1) to a hexadecimal color code.
    
    Parameters:
    rgb_tuple (tuple): A tuple of three float values between 0 and 1, representing the RGB values.
    
    Returns:
    str: The hexadecimal color code.
    """
    r, g, b = [int(255 * c) for c in rgb_tuple]
    return f"#{r:02X}{g:02X}{b:02X}"

def plot_stacked_barchart(str_list, data_list, filename, 
                          bins=[-0.05, 0.25, 0.55, 0.75, 0.85, 0.95, 1.05],
                          bar_width=0.8,
                          N=742,
                          out_folder='figs/',
                          shades=None,
                          xfontsize=14):
    hist_list = []
    for data in data_list:
        hist, _ = np.histogram(data, bins)
        hist_list.append(hist)

    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plt.subplots()
    
    n_bars = len(str_list)

    # Positions of the bars on the x-axis
    bar_positions = np.arange(n_bars)

    # Stacking the bars for each range
    for i in range(len(bins) - 1):
        # Heights of the bars for this range
        bar_heights = [hist[i]/len(data) for hist, data in zip(hist_list, data_list)]
        print(bar_heights)
        # The bottom position for the bars
        if i == 0:
            bottoms = np.zeros(n_bars)
        else:
            bottoms += bar_heights_previous
        # Plot the bars
        ax.bar(bar_positions, bar_heights, bar_width, bottom=bottoms, label=f'({bins[i]:.1f},{bins[i+1]:.1f})', color=shades[i])
        bar_heights_previous = bar_heights

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(str_list, fontsize=xfontsize)
    # ax.legend(loc='best')
    plt.savefig(osp.join(out_folder, filename), bbox_inches='tight')
    

def read_data(filepath, books, metric):
    data = pickle.load(open(filepath, 'rb'))
    x = []
    for book in books:
        x.append(data[book][metric])
    return np.array(x)

def read_distr_data(filepath):
    data = pickle.load(open(filepath, 'rb'))
    return data
    
def plot_custom_barcharts(l_list, title, xlow, xhigh, kde=False, 
                          xtitle='mean', bar_width=0.04, width=1.0, 
                          precision=1, drop=False, palette='Set2', 
                          xlabel='', ylabel='', xrange=0.2, out_folder='figs/',
                          N=742, ymax=1):
    l_list = [np.round(l, 6) for l in l_list]

    l_unique_list, l_count_list = [], []
    for l in l_list:
        l_unique, l_count = np.unique(l, return_counts=True)
        l_unique_list.append(l_unique)
        l_count_list.append(l_count)

    unique = list(np.unique([x for l in l_unique_list for x in l]))
    print(unique)

    x_list = []
    x_list.append(unique[0])
    for i in range(1, len(unique)):
        if unique[i] - x_list[-1] <= xrange:
            continue
        x_list.append(unique[i])
    if not np.isclose(unique[-1], x_list[-1]):
        x_list.append(unique[-1])

    bins = x_list

    hist_list = []
    for data in l_list:
        hist, _ = np.histogram(data, bins)
        hist_list.append(hist)

    print('bins', bins)
    print('hist_list', hist_list)

    bins = bins[:-1]
    data = {'Unique': np.array(bins)}
    for i in range(len(l_list)):
        data[f'L{i}'] = np.array(hist_list[i])/N

    print(data)

    # print(data)
    df = pd.DataFrame(data)
    df_long = pd.melt(df, id_vars=['Unique'], value_vars=[f'L{i}' for i in range(len(l_list))], var_name='List', value_name='Count')
    
    # Plotting
    
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(data=df_long, x='Unique', y='Count', hue='List', palette=palette, width=width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend('')
    plt.grid(axis='y') 
    plt.xlim(xlow, xhigh)
    plt.ylim(0, ymax)
    if precision == 0:
        unique_str = [f'{x:.0f}' for x in bins]
    elif precision == 1:
        unique_str = [f'{x:.1f}' for x in bins]
    elif precision == 2:
        unique_str = [f'{x:.2f}' for x in bins]
    plt.grid(axis='y')
        
    print(unique_str)
        
    bar_plot.set_xticklabels(unique_str)
    
    plt.savefig(osp.join(out_folder, title), bbox_inches='tight')
    
def plot_kdeplot(data_list, title, colors, str_list=None, out_folder='figs/', alpha=0.1, legend=False):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    if legend:
        for data, st, color in zip(data_list, str_list, colors):
            # TODO: add the label
            sns.kdeplot(data=data, fill=True, ax=ax, clip=(0, 10), alpha=alpha, color=color, linewidth=2, label=st)
        plt.legend(fontsize=16, loc='upper left')
    else:
        for data, color in zip(data_list, colors):
            sns.kdeplot(data=data, fill=True, ax=ax, clip=(0, 10), alpha=alpha, color=color, linewidth=2)        
    plt.ylabel('')
    plt.xlim(-0.2, 4.2)
    plt.savefig(osp.join(out_folder, title), bbox_inches='tight')
    
    
def plot_grouped_barchart(str_list, data_list, title, palette, group_list=None, out_folder='figs/', xmax=0.27, figsize=(6,8), noyticks=True, legend=False):
    print(data_list)
    sns.set_style("whitegrid")
    d = {}
    d['Group'] = str_list
    for i, data in enumerate(data_list):
        if group_list:
            d[group_list[i]] = data
        else:
            d[f'List{i}'] = data
    df = pd.DataFrame(d)
    # Melt the DataFrame for easier plotting
    df_melted = df.melt(id_vars='Group', var_name='List', value_name='Value')

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted, x='Value', y='Group', hue='List', orient='h',
                palette=palette)
    if legend:
        plt.legend(fontsize=18)
    else:
        plt.legend('')
    plt.xlabel('')
    plt.ylabel('')
    if noyticks:
        plt.yticks([])
    plt.tight_layout()
    plt.xlim(0, xmax)
    plt.savefig(osp.join(out_folder, title), bbox_inches='tight')


def get_count_indexed_topics(uni_src_sel, uni_gen, cnt_gen):
    cnt_gen_sel = []
    for x in uni_src_sel:
        loc = np.where(uni_gen == x)[0]
        if len(loc) == 0:
            cnt_gen_sel.append(0)
        else:
            cnt_gen_sel.append(cnt_gen[loc][0])
    return cnt_gen_sel