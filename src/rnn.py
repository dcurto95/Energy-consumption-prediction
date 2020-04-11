from keras import Sequential
from keras.layers import LSTM, GRU, Dense, SimpleRNN, Dropout
from keras.optimizers import RMSprop


def create_model(input_shape, neurons, neurons_increase, drop, rnn_layers, dense_layers, activation, activation_r,
                 rnntype):
    """
    RNN architecture

    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else SimpleRNN
    RNN = GRU if rnntype == 'GRU' else RNN

    model = Sequential()
    if rnn_layers == 1:
        model.add(RNN(neurons, input_shape=input_shape, recurrent_dropout=drop, activation=activation))
    else:
        model.add(
            RNN(neurons, input_shape=input_shape, recurrent_dropout=drop, activation=activation, return_sequences=True))
        i = 1
        for i in range(1, rnn_layers - 1):
            model.add(RNN(neurons * i * neurons_increase, recurrent_dropout=drop, activation=activation,
                          return_sequences=True))
        model.add(RNN(neurons * i + 1 * neurons_increase, recurrent_dropout=drop, activation=activation))
    for i in range(1, dense_layers - 1):
        model.add(Dense((neurons * (dense_layers - 1 - i)) // neurons_increase))

    model.add(Dense(1))
    model.summary()

    return model


def kaggle_model(input_shape):
    rnn_model = Sequential()

    rnn_model.add(SimpleRNN(40, activation="tanh", return_sequences=True, input_shape=input_shape))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40, activation="tanh", return_sequences=True))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40, activation="tanh", return_sequences=False))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(Dense(1))

    rnn_model.summary()
    return rnn_model


def compile(model, optimizer, lr=None):
    if optimizer == 'rmsprop':
        if lr:
            optimizer = RMSprop(lr=lr)
        else:
            optimizer = RMSprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)


def fit(model, train_x, train_y, batch_size, epochs, validation_x, validation_y, verbose=0):
    return model.fit(train_x, train_y, batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(validation_x, validation_y),
                     verbose=verbose)


def evaluate(model, test_x, test_y, batch_size):
    return model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
