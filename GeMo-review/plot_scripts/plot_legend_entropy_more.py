import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import seaborn as sns

# Define the colors and text labels
# colors = ['#222222', '#444444', '#666666', '#888888', '#AAAAAA', '#CCCCCC']

palette_set2 = sns.color_palette("Set2")
palette_tab10 = sns.color_palette("tab10")

colors2 = palette_set2[:5]
colors1 = palette_set2[:1] + palette_tab10[1:5]

labels1 = [
    'src', 
    '$T=0.5$',
    '$T=0.8$',
    '$T=1.0$',
    '$T=1.2$',
]

labels2 = [
    'src',
    '$p=0.90$',
    '$p=0.95$',
    '$p=0.98$',
    '$p=1.00$',
]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 1))

# Turn off axis and set transparent background
ax.set_axis_off()
fig.patch.set_visible(False)

# Create a list of rectangle markers
markers = [mpatches.Rectangle((0, 0), 10, 10, facecolor=color, edgecolor='none') for color in colors1]

# Create the legend
legend = ax.legend(markers, labels1, loc='center', ncol=5, frameon=False,
                   handler_map={mpatches.Rectangle: HandlerPatch(patch_func=None)},
                   handlelength=0.8, handletextpad=0.4)

# Adjust the legend layout
plt.setp(legend.get_texts(), ha='center', va='center')

# Display the legend
plt.savefig('figs/fig_legend_entropy_more_1.pdf', bbox_inches='tight')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 1))

# Turn off axis and set transparent background
ax.set_axis_off()
fig.patch.set_visible(False)

# Create a list of rectangle markers
markers = [mpatches.Rectangle((0, 0), 10, 10, facecolor=color, edgecolor='none') for color in colors2]

# Create the legend
legend = ax.legend(markers, labels2, loc='center', ncol=5, frameon=False,
                   handler_map={mpatches.Rectangle: HandlerPatch(patch_func=None)},
                   handlelength=0.8, handletextpad=0.4)

# Adjust the legend layout
plt.setp(legend.get_texts(), ha='center', va='center')

# Display the legend
plt.savefig('figs/fig_legend_entropy_more_2.pdf', bbox_inches='tight')