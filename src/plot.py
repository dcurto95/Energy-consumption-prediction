import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import numpy as np


def multiple_line_plot(x_list, y_list, labels, file_name, folder='.', title=''):
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
    if len(x_list[0]) > 10:
        indices = np.arange(0, len(x_list[0]), len(x_list[0]) // 10)
    else:
        indices = np.arange(0, len(x_list[0]))
    ax.set_xticks(np.asarray(x_list[0])[indices].tolist())
    ax.set_xticklabels(np.asarray(x_list[0])[indices].tolist(), rotation=20)

    ax.set_title(title)
    ax.legend()

    plt.savefig(
        ("../logs/" + folder + "/" + file_name + ".png"))
    plt.close()


def draw_history(history, folder, test_name):
    # list all data in history
    print(history.history.keys())

    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.grid(True)
    plt.savefig('../logs/' + folder + '/Loss_' + test_name + '.jpg')

    plt.close('all')


def plot_consumption(dataframe):
    dataframe.plot(figsize=(16, 4), legend=True)

    plt.title('DOM hourly power consumption data ')

    plt.show()
