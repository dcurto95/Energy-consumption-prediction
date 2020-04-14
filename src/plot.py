import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import  seaborn as sns
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


def plot_average_consumption(dataframe, column):
    plt.figure(figsize=(15, 6))

    plt.show()

def boxplot_hour_consumption(dataframe):
    _= dataframe.pivot_table(index=dataframe['Hour'],
                       columns='Weekday',
                       values='PJME_MW',
                       aggfunc='sum').plot(figsize=(15, 4),
                                           title='PJM East Region - Daily Trend')


def consumption_distribution_quarter(df):
    Q1 = df[df["Quarter"] == 1]
    Q2 = df[df["Quarter"] == 2]
    Q3 = df[df["Quarter"] == 3]
    Q4 = df[df["Quarter"] == 4]

    fig, axes = plt.subplots(2, 2, figsize=(17, 7), sharex=True, sharey=True)

    sns.distplot(Q1["PJME_MW"], color="skyblue", ax=axes[0, 0]).set_title("Quarter 1 - Consumption")
    sns.distplot(Q2["PJME_MW"], color="red", ax=axes[0, 1]).set_title("Quarter 2 - Consumption")
    sns.distplot(Q3["PJME_MW"], color="green", ax=axes[1, 0]).set_title("Quarter 3 - Consumption")
    sns.distplot(Q4["PJME_MW"], color="gray", ax=axes[1, 1]).set_title("Quarter 4 - Consumption")


def consumption_distribution_hour(df):

    modified= df.pivot_table(index=df['Hour'],
                       columns='Hour',
                       values='PJME_MW',
                       aggfunc='sum')
    plt.figure(figsize=(17, 7))
    sns.distplot(modified["PJME_MW"])


def plot_real_and_prediction(dataframe):

    gt = dataframe["RealMW"]
    pred = dataframe["PredictionMW"]

    gt = [v[0] for v in list(gt)]
    pred = [v[0] for v in list(pred)]

    plt.figure(figsize=(15, 6))
    plt.plot(gt, 'b', linewidth=2.5, linestyle="-", label='real')
    plt.plot(pred, 'r', linewidth=2.5, linestyle="-", label='prediction')
    plt.legend(loc='best')


def plot_best_worst_day(best,worst, df):

    year_best, month_best, day_best = extract_year_month_day(list(best.name))
    year_worst, month_worst, day_worst = extract_year_month_day(list(worst.name))

    worst_df = df.loc[(df['Year'] == year_worst) & (df['Month'] == month_worst) & (df['Day'] == day_worst)]
    best_df = df[(df['Year'] == year_best) & (df['Month'] == month_best) & (df['Day'] == day_best)]

    plot_real_and_prediction(worst_df)
    plot_real_and_prediction(best_df)

    plt.show()

def extract_year_month_day(value):
    return value[0], value[1], value[2]


def data_discovery(dataframe):
    consumption_distribution_quarter(dataframe)
    boxplot_hour_consumption(dataframe)
    consumption_distribution_hour(dataframe)
    plt.show()

