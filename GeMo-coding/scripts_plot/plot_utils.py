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
        bar_heights = [hist[i]/N for hist in hist_list]
        # The bottom position for the bars
        if i == 0:
            bottoms = np.zeros(n_bars)
        else:
            bottoms += bar_heights_previous
        # Plot the bars
        ax.bar(bar_positions, bar_heights, bar_width, bottom=bottoms, label=f'({bins[i]:.1f},{bins[i+1]:.1f})', color=shades[i])
        bar_heights_previous = bar_heights

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(str_list, fontsize=16)
    # ax.legend(loc='best')
    plt.savefig(osp.join(out_folder, filename), bbox_inches='tight')
    

def plot_barchart_any(l_list, title, xlow, xhigh, kde=False, xtitle='mean', bar_width=0.04, width=1.0, group=None, precision=1, drop=False, palette='Set2', xlabel='', ylabel='', out_folder='figs/', figsize=(10, 6), N=100):
    l_list = [np.round(l, 6) for l in l_list]

    l_unique_list, l_count_list = [], []
    for l in l_list:
        l_unique, l_count = np.unique(l, return_counts=True)
        l_unique_list.append(l_unique)
        l_count_list.append(l_count)

    unique = list(np.unique([x for l in l_unique_list for x in l]))
    print(unique)

    print('group', group)

    if group:
        st = unique[0]
        bins = []
        lt = [[] for i in range(len(l_list))]
        xi = [l_count[0] for l_count in l_count_list]
        n_bins = int((unique[-1] - unique[0]) / group) + 2
        st = int(unique[0])
        print('n_bins', n_bins)
        for i in range(n_bins):
            ed = st + group
            cnti = [len(np.where(np.logical_and(l >= st, l < ed))[0]) for l in l_list]
            
            if not drop or max(cnti) > 0:
                bins.append(st)
                for cnt, l in zip(cnti, lt):
                    l.append(cnt)
            st = ed
        unique = np.array(bins)
        print(unique, lt)

    else:
        lt = [[0] * len(unique) for l in l_list]
        for lit, l_unique, l_count in zip(lt, l_unique_list, l_count_list):
            for x, y in zip(l_unique, l_count):
                lit[unique.index(x)] = y
        print(lt)

        unique = np.array(unique)
        
    for i in range(len(l_list)):
        lt[i] = np.array(lt[i]) / N        

    data = {'Unique': unique}
    for i in range(len(l_list)):
        data[f'L{i}'] = lt[i]

    # print(data)
    df = pd.DataFrame(data)
    df_long = pd.melt(df, id_vars=['Unique'], value_vars=[f'L{i}' for i in range(len(l_list))], var_name='List', value_name='Count')
    
    # Plotting
    
    
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(data=df_long, x='Unique', y='Count', hue='List', palette=palette, width=width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend('')
    plt.grid(axis='y') 
    plt.xlim(xlow, xhigh)
    if precision == 0:
        unique_str = [f'{x:.0f}' for x in unique]
    elif precision == 1:
        unique_str = [f'{x:.1f}' for x in unique]
    elif precision == 2:
        unique_str = [f'{x:.2f}' for x in unique]
    plt.grid(axis='y')
        
    bar_plot.set_xticklabels(unique_str)
    # for i in range(1, 12, 2):
    #     unique_str[i] = ''
    # for i in range(1, 5): unique_str[i] = ''
    # for i in range(6, 10): unique_str[i] = ''
    # unique_str[-1] = ''
    print(unique_str)
        
    bar_plot.set_xticklabels(unique_str)
    
    plt.savefig(osp.join(out_folder, title), bbox_inches='tight')
    
def plot_grouped_barchart(str_list, data_list, title, palette, out_folder='figs/', xmax=0.27, figsize=(6,8), noyticks=True):
    print(data_list)
    sns.set_style("whitegrid")
    d = {}
    d['Group'] = str_list
    for i, data in enumerate(data_list):
        d[f'List{i}'] = data
    df = pd.DataFrame(d)
    # Melt the DataFrame for easier plotting
    df_melted = df.melt(id_vars='Group', var_name='List', value_name='Value')

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted, x='Value', y='Group', hue='List', orient='h',
                palette=palette)
    plt.legend('')
    plt.xlabel('')
    plt.ylabel('')
    plt.locator_params(axis='x', nbins=4)
    if noyticks:
        plt.yticks([])
    plt.tight_layout()
    plt.xlim(0, xmax)
    plt.savefig(osp.join(out_folder, title), bbox_inches='tight')

def plot_three_hist(l1, l2, l3, filename='default', out_folder='figs', ylabel='', loc='upper left', bins=-1, colors=sns.color_palette("Paired"), mode='time'):
    fig, ax = plt.subplots()
    
    plt.figure(figsize=(8, 6))
    if mode == 'time':
        l1 = [x for x in l1 if x < 200]
        l2 = [x for x in l2 if x < 200]
        l3 = [x for x in l3 if x < 200]
    else:
        l1 = [x for x in l1 if x < 30000]
        l2 = [x for x in l2 if x < 30000]
        l3 = [x for x in l3 if x < 30000]
    
    if bins > 0:
        sns.histplot(l1, color=colors[0], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
        sns.histplot(l2, color=colors[1], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
        sns.histplot(l3, color=colors[2], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
    else:
        sns.histplot(l1, color=colors[0], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        sns.histplot(l2, color=colors[1], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        sns.histplot(l3, color=colors[2], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        
    # plt.legend(fontsize=24, loc=loc)
    # plt.xlabel('Accuracy')
    # plt.xlim(10, 200)
    plt.ylabel(ylabel)
    # ax.set_xticklabels([10,50,150], fontsize=24)
    plt.savefig(osp.join(out_folder, filename), bbox_inches='tight')
    plt.show()
    
def plot_five_hist(l1, l2, l3, l4, l5, filename='default', out_folder='figs', ylabel='', loc='upper left', bins=-1, colors=sns.color_palette("Paired"), mode='time'):
    fig, ax = plt.subplots()
    
    plt.figure(figsize=(8, 6))
    if mode == 'time':
        l1 = [x for x in l1 if x < 200]
        l2 = [x for x in l2 if x < 200]
        l3 = [x for x in l3 if x < 200]
        l4 = [x for x in l4 if x < 200]
        l5 = [x for x in l5 if x < 200]
    else:
        l1 = [x for x in l1 if x < 30000]
        l2 = [x for x in l2 if x < 30000]
        l3 = [x for x in l3 if x < 30000]
        l4 = [x for x in l4 if x < 30000]
        l5 = [x for x in l5 if x < 30000]
    
    if bins > 0:
        sns.histplot(l1, color=colors[0], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
        sns.histplot(l2, color=colors[1], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
        sns.histplot(l3, color=colors[2], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
        sns.histplot(l4, color=colors[3], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
        sns.histplot(l5, color=colors[4], element='step', multiple='stack', bins=bins, stat='probability', alpha=0.3, linewidth=0.5)
    else:
        sns.histplot(l1, color=colors[0], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        sns.histplot(l2, color=colors[1], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        sns.histplot(l3, color=colors[2], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        sns.histplot(l4, color=colors[3], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        sns.histplot(l5, color=colors[4], element='step', stat='probability', alpha=0.1, linewidth=0.5)
        
    # plt.legend(fontsize=24, loc=loc)
    # plt.xlabel('Accuracy')
    # plt.xlim(10, 200)
    plt.ylabel(ylabel)
    # ax.set_xticklabels([10,50,150], fontsize=24)
    plt.savefig(osp.join(out_folder, filename), bbox_inches='tight')
    plt.show()