import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


if __name__ == "__main__" :

    slice_point = 10*288
    random_interval = random.randint(5,10)
    look_back = 200

    dataset = pd.read_csv(os.getcwd() + '/dataset/final_dataset.csv')

    training_df = dataset.iloc[:,[1,4,12,13,14,16,17,18]]
    training_df = training_df.assign(Total_t=training_df[['Coal','Nuclear','Natural gas','Batteries','Imports','Other']].sum(axis=1))
    training_df = training_df.fillna(0)

    complete_set = training_df[['Total_t']].values
    complete_set = complete_set.astype('float32')

    scaler = MinMaxScaler(feature_range = (0, 1))
    complete_set = scaler.fit_transform(complete_set)

    train = complete_set[:slice_point]
    validation = complete_set[random_interval*slice_point:random_interval*slice_point+(5*288)]
    print('Training set shape --> {}, \tValidation set shape --> {}'.format(train.shape, validation.shape))

    train = pd.DataFrame(train, columns=['Total (t)'])
    validation = pd.DataFrame(validation, columns=['Total (t)'])

    train = train.to_numpy()
    validation = validation.to_numpy()

    train_X, train_Y = create_dataset(train, look_back)
    validation_X, validation_Y = create_dataset(validation, look_back)

    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    validation_X = np.reshape(validation_X, (validation_X.shape[0], 1, validation_X.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['Accuracy'])
    model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)
    model.save("LSTM_model.h5")
    trainPredict = model.predict(train_X)
    trainPredict = scaler.inverse_transform(trainPredict)
    train_Y = scaler.inverse_transform([train_Y])

    validationPredict = model.predict(validation_X)
    validationPredict = scaler.inverse_transform(validationPredict)
    validation_Y = scaler.inverse_transform([validation_Y])

    trainScore = np.sqrt(mean_squared_error(train_Y[0], trainPredict[:,0]))
    validationScore = np.sqrt(mean_squared_error(validation_Y[0], validationPredict[:,0]))
    complete_set = complete_set[:slice_point+(5*288)]
    
    trainPredictPlot = np.empty_like(complete_set)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back: len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(complete_set)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1: len(complete_set) - 1, :] = validationPredict
    # # plot baseline and predictions
    plt.figure(figsize=(20,9))
    plt.title('Average Fossil Fuel Need -- Five minute ticks')
    plt.xlabel('5 minute ticks')
    plt.ylabel('Average Fossil Fuel Need')
    plt.plot(scaler.inverse_transform(complete_set))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.legend(['Complete_set', 'Training_Set_Predictions', 'Validation_Set_Predictions'])
    plt.show()