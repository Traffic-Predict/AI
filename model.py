# model.py

import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import SGD
from keras import callbacks
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import subprocess
# Normalized data 불러 오기
# feature_data.append([x_train_j, y_train_j, x_test_j, y_test_j])
# [x학습 데이터, y학습 데이터, x테스트 데이터, y테스트 데이터]가 리스트로 연결되어 있음
from data_preprocessing import normalized_data
from data_preprocessing import colors


def GRU_model(X_Train, y_Train, X_Test, save=True):
    early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)
    # callback delta 0.01 may interrupt the learning, could eliminate this step, but meh!

    # The GRU model
    model = Sequential()
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    # model.add(GRU(units=50, return_sequences=True,  input_shape=(X_Train.shape[1],1),activation='tanh'))
    # model.add(Dropout(0.2))
    model.add(GRU(units=50, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    # Compiling the model
    model.compile(optimizer=SGD(decay=1e-7, momentum=0.9), loss='mean_squared_error')
    model.fit(X_Train, y_Train, epochs=50, batch_size=150, callbacks=[early_stopping])
    pred_GRU = model.predict(X_Test)
    if save:
        now = datetime.datetime.now()
        filename = now.strftime("%Y%m%d-%H%M%S")
        filename = filename + ".h5"
        model.save(filename)
    return pred_GRU


# To calculate the root mean squred error in predictions
def RMSE_Value(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse


# To plot the comparitive plot of targets and predictions
def PredictionsPlot(test, predicted, m):
    plt.figure(figsize=(12, 5), facecolor="#627D78")
    plt.plot(test, color=colors[m], label="True Value", alpha=0.5)
    plt.plot(predicted, color="#627D78", label="Predicted Values")
    plt.title("GRU Traffic Prediction Vs True values")
    plt.xlabel("DateTime")
    plt.ylabel("Number of Vehicles")
    plt.legend()
    plt.show()


# 메인 함수
if __name__ == "__main__":
    subprocess.run(["python3", "data_preprocessing.py"])
    predict_lst = []
    for idx, data in enumerate(normalized_data):
        # 0 : x_train_j, 1 : y_train_j, 2 : x_test_j
        predict_data = GRU_model(data[0], data[1], data[2], True)
        predict_lst.append(predict_data)
        # 3 : y_test_j
        rmse_data = RMSE_Value(data[3], predict_data)
        PredictionsPlot(data[3], predict_data, idx)
        print(f"{idx}번 째 데이터 계산 완료")

    plt.show()
    print(f"predict completed")