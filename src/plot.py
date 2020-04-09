import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection

import numpy as np
def multiple_line_plot(x_list, y_list, labels, test_name, title=''):
    # plot the data
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    x_event_list = []
    y_event_list = []

    for x, y, color, label in zip(x_list, y_list, mcolors.TABLEAU_COLORS, labels):
        ax.plot(x, y, color=color, label=label)

    # add the events to the axis
    for x_event, y_event in zip(x_event_list, y_event_list):
        ax.add_collection(x_event)
        ax.add_collection(y_event)
    indices = np.arange(0, len(x_list[0]), len(x_list[0])//10)
    ax.set_xticks(np.asarray(x_list[0])[indices].tolist())
    ax.set_xticklabels(np.asarray(x_list[0])[indices].tolist(), rotation=20)

    ax.set_title(title)
    ax.legend()

    plt.savefig(
        ("../logs/" + test_name + "-pred_vs_truth.png"))
    plt.close()
