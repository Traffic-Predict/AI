# data_preprocessing.py
# Data Transformation And Preprocessing
# 2024-05-13
# created by totalcream

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller

def load_data(filepath):
    loaded_data = pd.read_csv(filepath)
    loaded_data.head()
    loaded_data.info()
    return loaded_data


def data_exploration(data_exploration):
    # 데이터 전처리 작업
    data_exploration["DateTime"] = pd.to_datetime(data_exploration["DateTime"])
    data_exploration = data_exploration.drop(["ID"], axis=1)  # dropping IDs
    return data_exploration



def preprocessing(preprocess_data):
    df_J = preprocess_data.pivot(columns="Junction", index="DateTime")
    df_J.describe()
    df_J.info()
    # Creating new sets
    df_1 = df_J[[('Vehicles', 1)]]
    df_2 = df_J[[('Vehicles', 2)]]
    df_3 = df_J[[('Vehicles', 3)]]
    df_4 = df_J[[('Vehicles', 4)]]
    df_4 = df_4.dropna()  # Junction 4 has limited data only for a few months

    # Dropping level one in dfs's index as it is a multi index data frame
    list_dfs = [df_1, df_2, df_3, df_4]
    for i in list_dfs:
        i.columns = i.columns.droplevel(level=1)

        # Function to plot comparitive plots of dataframes

    def Sub_Plots4(df_1, df_2, df_3, df_4, title):
        fig, axes = plt.subplots(4, 1, figsize=(15, 8), facecolor="#627D78", sharey=True)
        fig.suptitle(title)
        # J1
        pl_1 = sns.lineplot(ax=axes[0], data=df_1, color=colors[0])
        # pl_1=plt.ylabel()
        axes[0].set(ylabel="Junction 1")
        # J2
        pl_2 = sns.lineplot(ax=axes[1], data=df_2, color=colors[1])
        axes[1].set(ylabel="Junction 2")
        # J3
        pl_3 = sns.lineplot(ax=axes[2], data=df_3, color=colors[2])
        axes[2].set(ylabel="Junction 3")
        # J4
        pl_4 = sns.lineplot(ax=axes[3], data=df_4, color=colors[3])
        axes[3].set(ylabel="Junction 4")

    # Plotting the dataframe to check for stationarity
    Sub_Plots4(df_1.Vehicles, df_2.Vehicles, df_3.Vehicles, df_4.Vehicles, "Dataframes Before Transformation")
    return list_dfs


def visualize_timeseries(data):
    # 시계열 데이터 시각화 작업
    df = data.copy()
    colors = ["#FFD4DB", "#BBE7FE", "#D3B5E5", "#dfe2b6"]
    plt.figure(figsize=(20, 4), facecolor="#627D78")
    Time_series = sns.lineplot(x=data['DateTime'], y="Vehicles", data=data, hue="Junction", palette=colors)
    Time_series.set_title("Traffic On Junctions Over Years")
    Time_series.set_ylabel("Number of Vehicles")
    Time_series.set_xlabel("Date")
    # plt.show()

    # Exploring more features
    df["Year"] = df['DateTime'].dt.year
    df["Month"] = df['DateTime'].dt.month
    df["Date_no"] = df['DateTime'].dt.day
    df["Hour"] = df['DateTime'].dt.hour
    df["Day"] = df.DateTime.dt.strftime("%A")
    df.head()
    return df


def normalize(data, col):
    average = data[col].mean()
    stdev = data[col].std()
    df_normalized = (data[col] - average) / stdev
    df_normalized = df_normalized.to_frame()
    return df_normalized, average, stdev


def difference(df, col, interval):
    diff = []
    for i in range(interval, len(df)):
        # value = df[col][i] - df[col][i - interval]
        value = df[col].iloc[i] - df[col].iloc[i - interval]
        diff.append(value)
    return diff


def normalize_data(datum, col, interval="week"):
    checked_dfs = []
    for data in datum:
        df_N, av_J, std_J = normalize(data, col)
        if interval == "week":
            Diff = difference(df_N, col=col, interval=(24 * 7))  # taking a week's diffrence
            df_N = df_N[24*7:]
        elif interval == "day":
            Diff = difference(df_N, col=col, interval=(24))  # taking a day's diffrence
            df_N = df_N[24:]
        elif interval == "hour":
            Diff = difference(df_N, col=col, interval=1)  # taking an hour's diffrence
            df_N = df_N[1:]
        df_N.columns = ["Norm"]
        df_N["Diff"] = Diff
        checked_dfs.append(df_N)

    for dfs in checked_dfs:
        print("\n")
        Stationary_check(dfs)

    for i in range(len(checked_dfs)):
        checked_dfs[i] = checked_dfs[i].dropna()

    feature_data = []
    for dfs in checked_dfs:
        j_train, j_test = split_data(dfs)
        x_train_j, y_train_j = tnf(j_train)
        x_test_j, y_test_j = tnf(j_test)
        x_train_j, y_train_j = featurefixshape(x_train_j, x_test_j)
        feature_data.append([x_train_j, y_train_j, x_test_j, y_test_j])

    return feature_data


def Stationary_check(df):
    df_cleaned = df.dropna().to_numpy().flatten()
    check = adfuller(df_cleaned)
    print(f"ADF Statistic: {check[0]}")
    print(f"p-value: {check[1]}")
    print("Critical Values:")
    for key, value in check[4].items():
        print('\t%s: %.3f' % (key, value))
    if check[0] > check[4]["1%"]:
        print("Time Series is Non-Stationary")
    else:
        print("Time Series is Stationary")



def split_data(df):
    training_size = int(len(df) * 0.90)
    data_len = len(df)
    train, test = df[0:training_size], df[training_size:data_len]
    train, test = train.values.reshape(-1, 1), test.values.reshape(-1, 1)
    return train, test

def tnf(df):
    end_len = len(df)
    X = []
    y = []
    steps = 32
    for i in range(steps, end_len):
        X.append(df[i - steps:i, 0])
        y.append(df[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y

def featurefixshape(train, test):
    train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    test = np.reshape(test, (test.shape[0], test.shape[1], 1))
    return train, test


# 메인 함수
if __name__ == "__main__":
    colors = ["#FFD4DB", "#BBE7FE", "#D3B5E5", "#dfe2b6"]
    file_path = "../traffic.csv"
    print('\033[38;5;208m' + f"Loading {file_path}" + '\033[0m')
    raw_data = load_data(file_path)
    print('\033[38;5;208m' + f"load data is completed" + '\033[0m')
    exploration_data = data_exploration(raw_data)
    print('\033[38;5;208m' + f"Data exploration is completed" + '\033[0m')
    pre_processed_data = preprocessing(exploration_data)
    print('\033[38;5;208m' + f"Preprocessing data is completed" + '\033[0m')
    normalized_data = normalize_data(pre_processed_data, "Vehicles", "week")
    print('\033[38;5;208m' + f"Normalize data is completed" + '\033[0m')
    # print(normalized_data)