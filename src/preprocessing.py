from datetime import datetime

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

