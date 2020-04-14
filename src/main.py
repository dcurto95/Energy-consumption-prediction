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
    dataframe['Datetime'] = pd.to_datetime(dataframe['Datetime'])
    # plot.plot_consumption(dataframe)
    dataframe.sort_values(by="Datetime", inplace=True)
    # plot.plot_consumption(dataframe)
    dataframe = preprocessing.extract_features_from_datetime(dataframe)

    scaler, data = preprocessing.normalize_minmax(dataframe)

    if type(config['tunning_parameter']['step']) is list:
        parameter_steps = config['tunning_parameter']['step']
    else:
        parameter_steps = np.arange(0, config['tunning_parameter']['max_value'], config['tunning_parameter']['step'])

    for loop, i in enumerate(parameter_steps):
        if i == 0:
            i = 1

        config[config['tunning_parameter']['from']][config['tunning_parameter']['name']] = i

        X, y = preprocessing.sequence_data(data, config['arch']['window_size'])
        # test_without_norm = dataframe.to_numpy()[int((dataframe.shape[0] - config['arch']['window_size']) * 0.9):, 1:]

        _, a, _, b = train_test_split(dataframe.to_numpy()[config['arch']['window_size']:, 1:],
                                      dataframe.to_numpy()[config['arch']['window_size']:, 0], test_size=0.2,
                                      shuffle=False)
        _, test_without_norm, _, _ = train_test_split(a, b, test_size=0.5, shuffle=False)

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
        validation_x, test_x, validation_y, test_y = train_test_split(test_x, test_y, test_size=0.5, shuffle=False)

        print('X_train.shape = ', train_x.shape)
        print('y_train.shape = ', train_y.shape)
        print('X_test.shape = ', test_x.shape)
        print('y_test.shape = ', test_y.shape)

        since = time.time()

        model = rnn.create_model((train_x.shape[1], train_x.shape[2]),
                                 neurons=config['arch']['neurons'],
                                 neurons_increase=config['arch']['neurons_increase'],
                                 drop=config['arch']['drop'],
                                 rnn_layers=config['arch']['rnn_layers'],
                                 dense_layers=config['arch']['dense_layers'],
                                 activation=config['arch']['activation'],
                                 activation_r=config['arch']['activation_r'],
                                 rnntype=config['arch']['rnn'])

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
            best_config = loop
        print()
        print('MSE test= ', score)
        print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))

        prediction = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)
        print("Predicted:", prediction)

        prediction = preprocessing.inverse_minmax(prediction, scaler)
        test_y = preprocessing.inverse_minmax(test_y, scaler)
        print('Real MSE =', mean_squared_error(test_y, prediction))

        r2test = r2_score(test_y, prediction)
        r2pers = r2_score(test_y[ahead:], test_y[0:-ahead])
        print('R2 test= ', r2test)
        print('R2 test persistence =', r2pers)

        print("\nExecution time:", time.time() - since, "s")
        errors.append((score, r2test, time.time() - since))

        # plot.plot_prediction(prediction, test_y)

        x_values = test_without_norm.astype(np.intc)
        x_values = [str(datetime.datetime(year=x[3], month=x[2], day=x[1], hour=x[0])) for x in x_values]

        plot.multiple_line_plot([x_values, x_values],
                                [test_y, prediction],
                                ['Truth', 'Prediction'],
                                config['tunning_parameter']['name'] + "=" + str(i) + "-pred_vs_truth",
                                folder=config['test_name'],
                                title="Prediction vs. Truth", figsize=(10, 10))
        plot.draw_history(history, config['test_name'], config['tunning_parameter']['name'] + "=" + str(i))

        print("\n\n", errors)
        print("Best:\nNum. Layers:", best_config, "\n", errors[best_config - 1])

        if type(config['tunning_parameter']['step']) is list:
            x_values = config['tunning_parameter']['step']
        else:
            x_values = np.arange(0, config['tunning_parameter']['max_value'], config['tunning_parameter']['step'])
            x_values[0] = 1

        # x_values = np.arange(0, config['tunning_parameter']['max_value'], config['tunning_parameter']['step'])

        np_errors = np.asarray(errors)
        plot.line_subplot([x_values[:loop + 1], x_values[:loop + 1], x_values[:loop + 1]],
                          [np_errors[:, 0], np_errors[:, 1], np_errors[:, 2]],
                          ['MSE', 'R2', 'Execution time'],
                          (1, 3),
                          config['test_name'] + "/Parameter=" + config['tunning_parameter']['name'] + "-Loss",
                          title="Metrics for parameter " + config['tunning_parameter']['name'],
                          figsize=(20, 10))
