import datetime
import time
from sys import argv

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
    since = time.time()
    fpath = '../hourly-energy-consumption/PJME_hourly.csv'

    dataframe = pd.read_csv(fpath)
    config = load_config_file(argv[1])

    preprocessing.fix_missing_values(dataframe)
    preprocessing.separate_datetime(dataframe)

    print(dataframe.head())
    data_x = dataframe.iloc[:, :-1].to_numpy()
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1], 1))
    data_y = dataframe.iloc[:, -1].to_numpy()

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)
    validation_x, test_x, validation_y, test_y = train_test_split(test_x, test_y, test_size=0.5, shuffle=False)

    # train_x, train_y = train[:, :-1, :], train[:, -1, :]
    # validation_x, validation_y = validation[:, :-1, :], validation[:, -1, :]
    # test_x, test_y = test[:, :-1, :], test[:, -1, :]

    model = rnn.create_model((train_x.shape[1], train_x.shape[2]),
                             neurons=config['arch']['neurons'],
                             drop=config['arch']['drop'],
                             nlayers=config['arch']['nlayers'],
                             activation=config['arch']['activation'],
                             activation_r=config['arch']['activation_r'],
                             rnntype=config['arch']['rnn'],
                             impl=2)

    optimizer = config['training']['optimizer']
    lr = config['training']['lrate']
    batch_size = config['training']['batch']
    epochs = config['training']['epochs']

    rnn.compile(model, optimizer, lr)

    rnn.fit(model, train_x, train_y, batch_size, epochs, validation_x, validation_y, verbose=1)

    score = rnn.evaluate(model, test_x, test_y, batch_size)

    ahead = 1

    print()
    print('MSE test= ', score)
    print('MSE test persistence =', mean_squared_error(test_y[ahead:], test_y[0:-ahead]))

    prediction = model.predict(test_x, batch_size=config['training']['batch'], verbose=0)
    print("Predicted:", prediction)

    r2test = r2_score(test_y, prediction)
    r2pers = r2_score(test_y[ahead:], test_y[0:-ahead])
    print('R2 test= ', r2test)
    print('R2 test persistence =', r2pers)

    print("\nExecution time:", time.time() - since)

    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1]))
    test_x = [str(datetime.datetime(year=x[0], month=x[1], day=x[2], hour=x[3], minute=x[4], second=x[5])) for x in
              test_x]
    plot.multiple_line_plot([test_x, test_x], [test_y, prediction], ['Truth', 'Prediction'], config['test_name'],
                            title="Prediction vs. Truth")
