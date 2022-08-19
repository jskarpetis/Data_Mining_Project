import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout

if __name__ == "__main__" : 
    # Types of non-renewable energy in our dataset 
    #   - Coal
    #   - Nuclear
    #   - Natural Gas
    #   - Batteries

    path = os.getcwd() + '/dataset/final_dataset.csv'
    dataset = pd.read_csv(path)

    # print(dataset.head)

    training_df = dataset.iloc[:,[1,4,12,13,14,16,17,18]]

    training_df['Total (t)'] = training_df[['Coal','Nuclear','Natural gas','Batteries','Imports','Other']].sum(axis=1)
    training_df['Total (t + 1)'] = training_df.iloc[1:,2:8].sum(axis=1)
    training_df['Total (t + 1)'] = training_df['Total (t + 1)'].shift(-1)

    training_df['Current demand (t + 1)'] = training_df.iloc[1:,1]
    training_df['Current demand (t + 1)'] = training_df['Current demand (t + 1)'].shift(-1)
    
    training_df = training_df.fillna(0)

    training_set = training_df[['Current demand','Current demand (t + 1)','Total (t)','Total (t + 1)']].values
    training_set = training_set.astype('float32')

    print(training_set)
    scaler = MinMaxScaler(feature_range = (0, 1))
    training_set = scaler.fit_transform(training_set)
    
    train_size = int(len(training_set)*0.67)
    test_size = len(training_set) - train_size
    
    train, test = training_set[0:train_size , :] , training_set[train_size:len(training_set), :]

    print(f" Train Size : {len(train)} \n Test Size: {len(test)} ")



    X_train = []
    y_train = []
    
    # For X_train we are gonna give the current demand ---> discussion about forecast
    # for y_train we are going to have total number of non renewable energy needed 

    # Neural Network
    # regressor = Sequential()
    # regressor.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1],1)))
    # regressor.Dropout(0.2)

    # regressor.add(Dense(units=1))

    # regressor.compile(optimizer='adam', loss='mean_squared_error')



