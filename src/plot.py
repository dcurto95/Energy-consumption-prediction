import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import numpy as np


def multiple_line_plot(x_list, y_list, labels, file_name, folder='.', title='', figsize=(20, 20)):
    # plot the data
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    for x, y, color, label in zip(x_list, y_list, mcolors.TABLEAU_COLORS, labels):
        ax.plot(x, y, color=color, label=label)

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


def plot_prediction(gt, pred):
    plt.figure(figsize=(15, 6))
    plt.plot(gt[0:1000], 'b', linewidth=2.5, linestyle="-", label='real')
    plt.plot(pred[0:1000], 'r', linewidth=2.5, linestyle="-", label='prediction')
    plt.legend(loc='best')
    plt.show()


def line_subplot(x_list, y_list, labels, subplots, file_name, folder='.', title='', figsize=(20, 20)):
    # plot the data
    fig = plt.figure(figsize=figsize)

    if len(x_list[0]) > 10:
        indices = np.arange(0, len(x_list[0]), len(x_list[0]) // 10)
    else:
        indices = np.arange(0, len(x_list[0]))

    for i, (x, y, color, label) in enumerate(zip(x_list, y_list, mcolors.TABLEAU_COLORS, labels)):
        ax = fig.add_subplot(subplots[0], subplots[1], i + 1)
        ax.plot(x, y, color=color)

        ax.set_xticks(np.asarray(x_list[0])[indices].tolist())
        ax.set_xticklabels(np.asarray(x_list[0])[indices].tolist())
        ax.grid(True)

        ax.set_title(label)
    plt.suptitle(title)
    plt.savefig(
        ("../logs/" + folder + "/" + file_name + ".png"))
    plt.close()
