import os
import joblib


def load_dataframes():
    dataframes = {}

    for file in os.listdir('./dataframes'):
        if file[-3:] == 'pkl' or file[-3:] == 'csv':
            df = joblib.load(os.path.join('./dataframes/', file))
            dataframes[file[:-4]] = df
    return dataframes
