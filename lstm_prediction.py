import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import os


def load_processed_data(directory, nodeid):
    filepath = os.path.join(directory, f'{nodeid}', f'{nodeid}.csv')
    print(f"Trying to load data from: {filepath}")  # 파일 경로 출력
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        print("Data loaded successfully.")
        print(data.head())  # 데이터프레임 구조 출력
        print(data.columns)  # 데이터프레임의 열 이름 출력
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


def plot_predictions(data, train_predict, test_predict, scaler, time_step):
    # 인덱스 설정
    data.set_index('datetime', inplace=True)
    data_values = data['speed'].values.reshape(-1, 1)

    # 실제 데이터와 예측 데이터의 인덱스 설정
    train_predict_plot = np.empty_like(data_values)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

    test_predict_plot = np.empty_like(data_values)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(data_values) - 1, :] = test_predict

    # 그래프 그리기
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
    directory = "sources/processed"
    checkpoint_dir = "sources/train/checkpoint"
    model_dir = "sources/model"
    nodeid = input("Enter nodeid: ")

    try:
        data = load_processed_data(directory, nodeid)

        # 'datetime' 열이 있는지 확인하고, 없으면 에러 발생
        if 'datetime' not in data.columns:
            raise KeyError("The 'datetime' column is not found in the data.")

        # Set the datetime column as index
        data.set_index('datetime', inplace=True)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['speed'] = scaler.fit_transform(data[['speed']])

        # Convert data to numpy array
        data_values = data['speed'].values
        data_values = data_values.reshape(-1, 1)

        # Create dataset for LSTM
        time_step = 10
        X, y = create_dataset(data_values, time_step)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        test_size = len(X) - train_size
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        y_train, y_test = y[0:train_size], y[train_size:len(y)]

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create the checkpoint directory if it doesn't exist
        checkpoint_path = os.path.join(checkpoint_dir, nodeid)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Create the model directory if it doesn't exist
        model_path = os.path.join(model_dir, nodeid)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Check if there is a saved model
        checkpoint_model_path = os.path.join(checkpoint_path, "model.keras")
        if os.path.exists(checkpoint_model_path):
            model = load_model(checkpoint_model_path)
            print("Loaded model from checkpoint.")
        else:
            # Create the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_model_path,
            save_weights_only=False,
            save_best_only=True,
            monitor='loss',
            verbose=1
        )

        # Train the model
        model.fit(X_train, y_train, batch_size=1, epochs=1, callbacks=[checkpoint_callback])

        # Save the final model
        final_model_path = os.path.join(model_path, "model.h5")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert predictions back to original scale
        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform([y_train])
        test_predict = scaler.inverse_transform(test_predict)
        y_test = scaler.inverse_transform([y_test])

        # Plot the results
        plot_predictions(data, train_predict, test_predict, scaler, time_step)

    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)


if __name__ == "__main__":
    main()
