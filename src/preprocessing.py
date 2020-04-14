from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def fix_missing_values(dataframe):
    dataframe.dropna(inplace=True)


def separate_datetime(dataframe):
    datetime_column = dataframe['Datetime']

    dataframe.insert(0, "second",
                     datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").second)))
    dataframe.insert(0, "minute",
                     datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)))
    dataframe.insert(0, "hour", datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)))

    dataframe.insert(0, "day", datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day)))
    dataframe.insert(0, "month", datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").month)))
    dataframe.insert(0, "year", datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year)))

    dataframe.drop(columns=['Datetime'], inplace=True)


def extract_features_from_datetime(dataframe):
    dataframe["Datetime"] = pd.to_datetime(dataframe["Datetime"])
    dataframe["Hour"] = dataframe["Datetime"].dt.hour
    dataframe["Day"] = dataframe["Datetime"].dt.day
    dataframe["Month"] = dataframe["Datetime"].dt.month
    dataframe["Year"] = dataframe["Datetime"].dt.year
    dataframe["Weekday"] = dataframe["Datetime"].dt.dayofweek
    dataframe["Quarter"] = dataframe["Datetime"].dt.quarter

    dataframe.drop(columns=["Datetime"], inplace=True)

    # columns = list(dataframe.columns)
    # columns.append(columns.pop(columns.index("PJME_MW")))
    # return dataframe.reindex(columns=columns)

    return dataframe


def normalize_zscore(dataframe):
    scaler = StandardScaler()
    dataframe["PJME_MW"] = scaler.fit_transform(dataframe["PJME_MW"].values.reshape(-1, 1))


def normalize_minmax(dataframe):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataframe.values)
    scaler.fit(dataframe.iloc[:, 0].values.reshape(-1, 1))
    # dataframe["PJME_MW"] = scaler.transform(dataframe.iloc[:, 0].values.reshape(-1, 1))
    return scaler, data


def inverse_scaling(data, scaler):
    data_rescaled = scaler.inverse_transform(data)
    return data_rescaled


def inverse_minmax(column, scaler):
    scale = MinMaxScaler()
    scale.min_, scale.scale_ = scaler.min_[0], scaler.scale_[0]
    rescaled = scale.inverse_transform(column)
    return rescaled

def sequence_data(data, seq_len):
    X = np.zeros((len(data) - seq_len, seq_len, data.shape[1]))
    y = np.zeros(len(data) - seq_len)

    for i in range(seq_len, len(data)):
        X[i - seq_len] = data[i - seq_len:i]
        y[i - seq_len] = data[i, 0]

    X = X.reshape((-1, seq_len, data.shape[1]))
    y = y.reshape((-1, 1))

    return X, y
