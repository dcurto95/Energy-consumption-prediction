import datetime
import os
import time
from shutil import copyfile
from sys import argv

import numpy as np
import pandas as pd
from pandas._libs import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import plot
import preprocessing
import rnn


def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


if __name__ == '__main__':
    best_error = float('inf')
    best_config = 0

    errors = []
    fpath = '../hourly-energy-consumption/PJME_hourly.csv'

    dataframe = pd.read_csv(fpath)
    config = load_config_file(argv[1], abspath=True)

    if not os.path.exists('../logs/' + config['test_name']):
        os.mkdir('../logs/' + config['test_name'])
    copyfile(argv[1], '../logs/' + config['test_name'] + "/config.json")

    preprocessing.fix_missing_values(dataframe)
    # Plot before normalizing data
    plot.plot_consumption(dataframe)
    preprocessing.normalize_data(dataframe)
    # Plot after normalizing data
    plot.plot_consumption(dataframe)
    # preprocessing.separate_datetime(dataframe)
    preprocessing.extract_features_from_datetime(dataframe)

    print(dataframe.head())
    data_x = dataframe.iloc[:, :-1].to_numpy()
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))
    data_y = dataframe.iloc[:, -1].to_numpy()
    data_y = data_y.reshape((data_y.shape[0], 1))

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)
    validation_x, test_x, validation_y, test_y = train_test_split(test_x, test_y, test_size=0.5, shuffle=False)

    for i in range(0, config['tunning_parameter']['max_value'], config['tunning_parameter']['step']):
        if i == 0:
            i = 1

        config[config['tunning_parameter']['from']][config['tunning_parameter']['name']] = i
        since = time.time()

        model = rnn.create_model((train_x.shape[1], train_x.shape[2]),
                                 neurons=config['arch']['neurons'],
                                 neurons_increase=config['arch']['neurons_increase'],
                                 drop=config['arch']['drop'],
                                 rnn_layers=config['arch']['rnn_layers'],
                                 dense_layers=config['arch']['dense_layers'],
                                 activation=config['arch']['activation'],
                                 activation_r=config['arch']['activation_r'],
                                 rnntype=config['arch']['rnn'],
                                 impl=2)

        optimizer = config['training']['optimizer']
        lr = config['training']['lrate']
        batch_size = config['training']['batch']
        epochs = config['training']['epochs']

        rnn.compile(model, optimizer, lr)

        history = rnn.fit(model, train_x, train_y, batch_size, epochs, validation_x, validation_y, verbose=1)

        score = rnn.evaluate(model, test_x, test_y, batch_size)

        ahead = 1

        if score < best_error:
            best_error = score
            best_config = i
        print()
        print('MSE test= ', score)
        print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))

        prediction = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)
        print("Predicted:", prediction)

        r2test = r2_score(test_y, prediction)
        r2pers = r2_score(test_y[ahead:], test_y[0:-ahead])
        print('R2 test= ', r2test)
        print('R2 test persistence =', r2pers)

        print("\nExecution time:", time.time() - since, "s")
        errors.append((score, r2test, time.time() - since))
        x_values = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        x_values = [str(datetime.datetime(year=x[0], month=x[1], day=x[2], hour=x[3], minute=x[4], second=x[5])) for x
                    in x_values]

        plot.multiple_line_plot([x_values, x_values],
                                [test_y, prediction],
                                ['Truth', 'Prediction'],
                                config['tunning_parameter']['name'] + "=" + str(i) + "-pred_vs_truth",
                                folder=config['test_name'],
                                title="Prediction vs. Truth")
        plot.draw_history(history, config['test_name'], config['tunning_parameter']['name'] + "=" + str(i))

    print("\n\n", errors)
    print("Best:\nNum. Layers:", best_config, "\n", errors[best_config - 1])

    x_values = np.arange(0, config['tunning_parameter']['max_value'], config['tunning_parameter']['step'])
    x_values[0] = 1

    errors = np.asarray(errors)
    plot.multiple_line_plot([x_values, x_values],
                            [errors[:, 0], errors[:, 1]],
                            ['MSE', 'R2'],
                            config['test_name'] + "/Parameter=" + config['tunning_parameter']['name'] + "-Loss",
                            title="Parameters configuration Loss")
