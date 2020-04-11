from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def fix_missing_values(dataframe):
    dataframe.dropna(inplace=True)


def separate_datetime(dataframe):
    datetime_column = dataframe['Datetime']

    dataframe.insert(0, "second", datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").second)))
    dataframe.insert(0, "minute", datetime_column.apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)))
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

    #dataframe["Dayofyear"] = dataframe["Datetime"].dt.dayofyear

    #dataframe["Weekofyear"] = dataframe["Datetime"].dt.weekofyear

    dataframe.drop(columns=["Datetime"], inplace=True)


def normalize_data(dataframe):
    scaler = StandardScaler()
    dataframe["PJME_MW"] = scaler.fit_transform(dataframe["PJME_MW"].values.reshape(-1,1))



