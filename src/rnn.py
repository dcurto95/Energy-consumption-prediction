from keras import Sequential
from keras.layers import LSTM, GRU, Dense
from keras.optimizers import RMSprop


def create_model(input_shape, neurons, drop, nlayers, activation, activation_r, rnntype, impl=1):
    """
    RNN architecture

    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=input_shape, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=input_shape, implementation=impl,
                      recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                      return_sequences=True))
        for _ in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                      recurrent_activation=activation_r, implementation=impl))

    model.add(Dense(1))
    model.summary()

    return model


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
