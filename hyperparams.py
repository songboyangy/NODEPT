import matplotlib.pyplot as plt
import numpy as np

# Sample data for the three datasets
aps_data = {
    'Observation Length': [4, 5, 6, 7],
    'msle': [1.0519, 1.0374, 1.0126, 0.988],
    'mape': [0.2138, 0.2233, 0.2262, 0.2306]
}
twitter_data = {
    'Observation Length': [1, 2, 3, 4],
    'msle': [3.0033, 2.5044, 2.5015, 2.3079],
    'mape': [0.3013, 0.2793, 0.3063, 0.2769]
}
weibo_data = {
    'Observation Length': [1, 2, 3, 4],
    'msle': [1.9687, 1.9342, 1.8729, 1.8512],
    'mape': [0.2610, 0.2593, 0.2568, 0.2513]
}

memory_size_data = {
    'aps': {
        'memory_size': [8,16,24,32],
        'msle': [1.0588, 1.0374, 1.0153, 1.0272],
        'mape': [0.2244, 0.2233, 0.2205, 0.2198]
    },
    'twitter': {
        'memory_size': [8,16,24,32],
        'msle': [2.6539, 2.5044, 2.5165, 2.5382],
        'mape': [0.2949, 0.2793, 0.2798, 0.2835]
    },
    'weibo': {
        'memory_size': [8,16,24,32],
        'msle': [2.5258, 1.9639, 1.9195, 1.9204],
        'mape': [0.3211, 0.261, 0.2492, 0.2555]
    }
}

# Function to create a subplot for a dataset
# def create_subplot(ax, data, title):
#     # Create bar plot for MSLE
#     ax.bar(np.arange(len(data['Observation Length'])), data['msle'], width=0.4, label='MSLE', color='orange', hatch='/')
#
#     # Create line plot for PCC
#     ax2 = ax.twinx()
#     ax2.plot(np.arange(len(data['Observation Length'])), data['mape'], label='MAPE', color='brown', marker='o')
#
#     # Set title and labels
#     ax.set_title(title)
#     ax.set_xticks(np.arange(len(data['Observation Length'])))
#     ax.set_xticklabels(data['Observation Length'])
#     ax.set_xlabel('Observation Length')
#     ax.set_ylabel('MSLE')
#     ax2.set_ylabel('MAPE')
#
#     # Set legend
#     ax.legend(loc='upper left')
#     ax2.legend(loc='upper right')
def create_subplot(ax, data, title, right_y_lim=None):
    # Create bar plot for MSLE with crossed hatch
    bars = ax.bar(np.arange(len(data['Observation Length'])), data['msle'], width=0.4, label='MSLE', color='orange',
                  hatch='x')

    # Create line plot for MAPE
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(data['Observation Length'])), data['mape'], label='MAPE', color='brown', marker='o')

    # Set larger x-tick labels for better visibility
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    # Set custom right y-axis limit if provided
    if right_y_lim:
        ax2.set_ylim(right_y_lim)

    # Set title and labels
    ax.set_title(title)
    ax.set_xticks(np.arange(len(data['Observation Length'])))
    ax.set_xticklabels(data['Observation Length'])
    ax.set_xlabel('Observation Length', fontsize=14)
    ax.set_ylabel('MSLE', fontsize=14)
    ax2.set_ylabel('MAPE', fontsize=14)

    # Set legend
    ax.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

bar_color = (242/255, 204/255, 142/255)
# Create a figure with subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))
#
# # Create subplots
# create_subplot(axs[0], aps_data, 'Performance on APS')
# create_subplot(axs[1], twitter_data, 'Performance on Twitter')
# create_subplot(axs[2], weibo_data, 'Performance on Weibo')
fig, axs = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)

# Right y-axis limits for each subplot to make line plot span appear smaller
right_y_lim_aps = (0.16, 0.24)
right_y_lim_twitter = (0.22, 0.32)
right_y_lim_weibo = (0.22, 0.27)

# Create subplots with the updated settings
create_subplot(axs[0], aps_data, 'Performance on APS', right_y_lim_aps)
create_subplot(axs[1], twitter_data, 'Performance on Twitter', right_y_lim_twitter)
create_subplot(axs[2], weibo_data, 'Performance on Weibo', right_y_lim_weibo)

# Adjust layout
plt.tight_layout()
plt.savefig('figure/Observation Length.png')
plt.show()
