import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
# Using the provided memory_size_data for the Memory Size plots

memory_size_data = {
    'aps': {
        'memory_size': [8, 16, 24, 32],
        'msle': [1.0588, 1.0374, 1.0153, 1.0272],
        'mape': [0.2244, 0.2233, 0.2205, 0.2198]
    },
    'twitter': {
        'memory_size': [8, 16, 24, 32],
        'msle': [2.6539, 2.5044, 2.5165, 2.5382],
        'mape': [0.2949, 0.2793, 0.2798, 0.2835]
    },
    'weibo': {
        'memory_size': [8, 16, 24, 32],
        'msle': [2.5258, 1.9639, 1.9195, 1.9204],
        'mape': [0.3211, 0.261, 0.2492, 0.2555]
    }
}
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
# Define colors for Observation Length and Memory Size
#observation_color = bar_color  # Already defined earlier (R: 242, G: 204, B: 142)
memory_color = (142/255, 168/255, 242/255)  # A shade of blue
y_lim={'twitter':{'left':(1,3.35),'right':(0.22, 0.32)},
       'Weibo':{'obs':{'left':(1.5,2),'right':(0.22, 0.27)},'mem': {'left':(1.5,2.7),'right':(0.22, 0.35)}                         },
       'APS':{'left':(0,1.3),'right':(0.16, 0.24)}}
# Define function to create a subplot for Observation Length
def create_observation_subplot(ax, data, title=None,y_lim=None):
    ax.bar(np.arange(len(data['Observation Length'])), data['msle'], width=0.4, label='MSLE', color='orange', hatch='x')
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(data['Observation Length'])), data['mape'], label='MAPE', color='brown', marker='o')
    ax.tick_params(axis='x', labelsize=18)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    #ax.set_title(title)
    ax.set_xticks(np.arange(len(data['Observation Length'])))
    ax.set_xticklabels(data['Observation Length'])
    if y_lim:
        ax.set_ylim(y_lim['left'])
        ax2.set_ylim(y_lim['right'])
    # ax.set_xlabel('Observation Length', fontsize=14)
    # ax.set_ylabel('MSLE', fontsize=14)
    # ax2.set_ylabel('MAPE', fontsize=14)
    # ax.legend(loc='upper left', fontsize=12)
    # ax2.legend(loc='upper right', fontsize=12)

# Define function to create a subplot for Memory Size
def create_memory_subplot(ax, data, title,y_lim=None):
    ax.bar(np.arange(len(data['memory_size'])), data['msle'], width=0.4, label='MSLE', color=memory_color, hatch='/')
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(data['memory_size'])), data['mape'], label='MAPE', color='brown', marker='o')
    ax.tick_params(axis='x', labelsize=18)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    #ax.set_title(title)
    ax.set_xticks(np.arange(len(data['memory_size'])))
    ax.set_xticklabels(data['memory_size'])
    if y_lim:
        ax.set_ylim(y_lim['left'])
        ax2.set_ylim(y_lim['right'])
    # ax.set_xlabel('Memory Size', fontsize=14)
    # ax.set_ylabel('MSLE', fontsize=14)
    # ax2.set_ylabel('MAPE', fontsize=14)
    # ax.legend(loc='upper left', fontsize=12)
    # ax2.legend(loc='upper right', fontsize=12)

# Create a 3x2 grid figure
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.3)


# Plot for Twitter
create_observation_subplot(axs[0, 0], twitter_data, 'Twitter (Observation Length)',y_lim=y_lim['twitter'])
create_memory_subplot(axs[0, 1], memory_size_data['twitter'], 'Twitter (Memory Size)',y_lim=y_lim['twitter'])

# Plot for Weibo
create_observation_subplot(axs[1, 0], weibo_data, 'Weibo (Observation Length)',y_lim=y_lim['Weibo']['obs'])
create_memory_subplot(axs[1, 1], memory_size_data['weibo'], 'Weibo (Memory Size)',y_lim=y_lim['Weibo']['mem'])

# Plot for APS
create_observation_subplot(axs[2, 0], aps_data, 'APS (Observation Length)',y_lim=y_lim['APS'])
create_memory_subplot(axs[2, 1], memory_size_data['aps'], 'APS (Memory Size)',y_lim=y_lim['APS'])
# Save the figure to a file and show it
plt.savefig('figure/hyperparams_figure1.png')
plt.show()
