import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from datetime import datetime, timedelta


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


def prepare_future_data(last_known_datetime, future_datetime, freq='5T'):
    steps = int((future_datetime - last_known_datetime) / timedelta(minutes=5))
    future_dates = [last_known_datetime + timedelta(minutes=5 * i) for i in range(1, steps + 1)]
    return pd.DataFrame({'datetime': future_dates})


def predict_future_speed(model, data, future_data, scaler, time_step=10):
    last_data = data[-time_step:].values
    future_predictions = []

    for _ in range(len(future_data)):
        input_data = last_data.reshape((1, time_step, 1))
        predicted_speed = model.predict(input_data)
        future_predictions.append(predicted_speed[0, 0])
        last_data = np.append(last_data[1:], predicted_speed[0, 0])

    future_data['predicted_speed'] = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_data


def plot_predictions(data, train_predict, test_predict, future_data, scaler, time_step, nodeid):
    data_values = data['speed'].values.reshape(-1, 1)

    print(f"train_predict shape: {train_predict.shape}")
    print(f"test_predict shape: {test_predict.shape}")
    print(f"data_values shape: {data_values.shape}")

    train_predict_plot = np.empty_like(data_values)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

    test_predict_plot = np.empty_like(data_values)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2):len(train_predict) + (time_step * 2) + len(test_predict),
    :] = test_predict

    print(f"train_predict_plot shape: {train_predict_plot.shape}")
    print(f"test_predict_plot shape: {test_predict_plot.shape}")

    plt.figure(figsize=(18, 8))
    plt.plot(data.index, scaler.inverse_transform(data_values), label='True Data')
    plt.plot(data.index, train_predict_plot, label='Train Predictions')
    plt.plot(data.index, test_predict_plot, label='Test Predictions')
    plt.plot(future_data['datetime'], future_data['predicted_speed'], label='Future Predictions', linestyle='--')
    plt.title('Speed Prediction Using LSTM')
    plt.xlabel('Datetime')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # TODO : 입력받은 시간까지 예측시간 출력하도록 수정
    # 현재 시간으로 파일 이름 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("sources/predict", nodeid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"prediction_{timestamp}.png")

    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    processed_dir = "sources/processed"
    model_dir = "sources/model"
    nodeid = input("Enter nodeid: ")
    future_datetime_str = input("Enter future datetime (YYYY-MM-DD HH:MM): ")
    future_datetime = datetime.strptime(future_datetime_str, "%Y-%m-%d %H:%M")

    try:
        data = load_processed_data(processed_dir, nodeid)

        if 'datetime' not in data.columns:
            raise KeyError("The 'datetime' column is not found in the data.")

        data.set_index('datetime', inplace=True)

        # 결측값 또는 비정상적인 값 처리
        if data['speed'].isnull().any():
            print("Data contains null values. Filling null values with the previous valid value.")
            data['speed'].fillna(method='ffill', inplace=True)
        if not np.isfinite(data['speed']).all():
            print("Data contains non-finite values. Replacing non-finite values with the previous valid value.")
            data['speed'].replace([np.inf, -np.inf], np.nan, inplace=True)
            data['speed'].fillna(method='ffill', inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data['speed'] = scaler.fit_transform(data[['speed']])

        data_values = data['speed'].values
        data_values = data_values.reshape(-1, 1)

        time_step = 10
        X, y = create_dataset(data_values, time_step)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

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

        # 예측 결과를 데이터프레임에 추가하여 비교
        data['train_predict'] = np.nan
        data['test_predict'] = np.nan

        train_predict_indices = data.iloc[time_step:len(train_predict) + time_step].index
        data.loc[train_predict_indices, 'train_predict'] = train_predict[:, 0]

        test_predict_indices = data.iloc[
                               len(train_predict) + (time_step * 2):len(train_predict) + (time_step * 2) + len(
                                   test_predict)].index
        if len(test_predict_indices) != len(test_predict):
            print(f"Length mismatch: {len(test_predict_indices)} indices, {len(test_predict)} predictions")
            test_predict = test_predict[:len(test_predict_indices)]
        data.loc[test_predict_indices, 'test_predict'] = test_predict[:, 0]

        # 미래 데이터 예측
        last_known_datetime = data.index[-1]
        future_data = prepare_future_data(last_known_datetime, future_datetime)  # 미래 시간까지 예측
        future_predictions = predict_future_speed(model, data['speed'], future_data, scaler, time_step)

        # 결과 출력
        target_prediction = future_predictions.loc[future_predictions['datetime'] == future_datetime]
        if not target_prediction.empty:
            predicted_speed = target_prediction['predicted_speed'].values[0]
            print(f"Predicted speed for {future_datetime} is {predicted_speed:.2f} km/h")
        else:
            print(f"No prediction available for {future_datetime}")

        # 실제 데이터와 예측 데이터를 비교하는 그래프를 저장
        plot_predictions(data, train_predict, test_predict, future_predictions, scaler, time_step, nodeid)

    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)
    except ValueError as e:
        print(f"ValueError: {e}")


if __name__ == "__main__":
    main()
