import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

# Define the colors and text labels
colors = ['#444444', '#666666', '#888888', '#AAAAAA', '#CCCCCC']
labels = ['[0,0.2]', '(0.2,0.4]', '(0.4,0.6]', '(0.6,0.8]', '[0.8,1.0]']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 1))

# Turn off axis and set transparent background
ax.set_axis_off()
fig.patch.set_visible(False)

# Create a list of rectangle markers
markers = [mpatches.Rectangle((0, 0), 10, 10, facecolor=color, edgecolor='none') for color in colors]

# Create the legend
legend = ax.legend(markers, labels, loc='center', ncol=len(colors), frameon=False,
                   handler_map={mpatches.Rectangle: HandlerPatch(patch_func=None)},
                   handlelength=0.8, handletextpad=0.4)

# Adjust the legend layout
plt.setp(legend.get_texts(), ha='center', va='center')

# Display the legend
plt.savefig('figs/legend_jaccard_fade.pdf', bbox_inches='tight')