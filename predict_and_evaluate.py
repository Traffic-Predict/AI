import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os


def load_processed_data(directory, nodeid):
    filepath = os.path.join(directory, f'{nodeid}', f'{nodeid}.csv')
    print(f"Trying to load data from: {filepath}")
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        print("Data loaded successfully.")
        print(data.head())  # 데이터프레임 구조 출력
        return data
    else:
        raise FileNotFoundError(f"No processed data found for nodeid {nodeid} in {directory}")


def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def plot_predictions(original_data, train_predict, test_predict, scaler, time_step):
    data = original_data.copy()  # 데이터 복사
    data.set_index('datetime', inplace=True)
    data_values = data['speed'].values.reshape(-1, 1)

    train_predict_plot = np.empty_like(data_values)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

    test_predict_plot = np.empty_like(data_values)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2):len(train_predict) + (time_step * 2) + len(test_predict),
    :] = test_predict

    plt.figure(figsize=(18, 8))
    plt.plot(data.index, scaler.inverse_transform(data_values), label='True Data')
    plt.plot(data.index, train_predict_plot, label='Train Predictions')
    plt.plot(data.index, test_predict_plot, label='Test Predictions')
    plt.title('Speed Prediction Using LSTM')
    plt.xlabel('Datetime')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    processed_dir = "sources/processed"
    model_dir = "sources/model"
    nodeid = input("Enter nodeid: ")

    try:
        data = load_processed_data(processed_dir, nodeid)

        if 'datetime' not in data.columns:
            raise KeyError("The 'datetime' column is not found in the data.")

        data.set_index('datetime', inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data['speed'] = scaler.fit_transform(data[['speed']])

        data_values = data['speed'].values
        data_values = data_values.reshape(-1, 1)

        time_step = 10
        X, y = create_dataset(data_values, time_step)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model_path = os.path.join(model_dir, nodeid, "model.h5")
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("Loaded model from", model_path)
        else:
            raise FileNotFoundError(f"No saved model found for nodeid {nodeid} in {model_path}")

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform([y_train])
        test_predict = scaler.inverse_transform(test_predict)
        y_test = scaler.inverse_transform([y_test])

        plot_predictions(data.reset_index(), train_predict, test_predict, scaler, time_step)

    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)
    except ValueError as e:
        print(f"ValueError: {e}")


if __name__ == "__main__":
    main()
