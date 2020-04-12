from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


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
    #data = scaler.fit_transform(dataframe.values)
    scaler.fit(dataframe.iloc[:, 0].values.reshape(-1, 1))
    dataframe["PJME_MW"] = scaler.transform(dataframe.iloc[:, 0].values.reshape(-1, 1))
    return scaler, dataframe


def inverse_minmax(column, scaler):
    scale = MinMaxScaler()
    scale.min_,scale.scale_ = scaler.min_[0], scaler.scale_[0]
    rescaled = scale.inverse_transform(column)
    return rescaled


def convert_data(dataframe):
    # Scaling the input data
    scaler = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = scaler.fit_transform(dataframe.values)
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(dataframe.iloc[:, 0].values.reshape(-1, 1))
    label_scalers = label_sc


    # Define lookback period and split inputs/labels
    lookback = 2
    inputs = np.zeros((len(data) - lookback, lookback, dataframe.shape[1]))
    labels = np.zeros(len(data) - lookback)

    for i in range(lookback, len(data)):
        inputs[i - lookback] = data[i - lookback:i]
        labels[i - lookback] = data[i, 0]
    inputs = inputs.reshape(-1, lookback, dataframe.shape[1])
    labels = labels.reshape(-1, 1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.1*len(inputs))

    train_x = inputs[:-test_portion]
    train_y = labels[:-test_portion]

    test_x = (inputs[-test_portion:])
    test_y = (labels[-test_portion:])


    return train_x,test_x,train_y,test_y