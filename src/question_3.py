import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Input, layers, Model
import pandas as pd


def keras_model(x, y):
    input = Input(shape=(len(x), len(x[0])), name='Input')
    hidden1 = layers.LSTM(units=4)(input)
    hidden2 = layers.Dense(units=3, activation='relu')(hidden1)
    output = layers.Dense(units=len(y[0]), activation='softmax')(hidden2)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    path = os.getcwd() + '/dataset/final_dataset.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.loc[:, ['Day ahead forecast', 'Hour ahead forecast', 'Current demand', 'Solar', 'Wind',
                              'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]

    x_train_full, x_valid_full = dataset[:250000], dataset[250000:]

    y_train = x_train_full.loc[:, ['Solar', 'Wind',
                                   'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]
    y_valid = x_valid_full.loc[:, ['Solar', 'Wind',
                                   'Geothermal', 'Biomass', 'Biogas', 'Small hydro', 'Coal', 'Nuclear', 'Natural gas', 'Large hydro', 'Batteries', 'Imports', 'Other']]

    x_train_full = x_train_full.loc[:, [
        'Day ahead forecast', 'Hour ahead forecast', 'Current demand']]

    x_valid_full = x_valid_full.loc[:, [
        'Day ahead forecast', 'Hour ahead forecast', 'Current demand']]

    # train_data = tf.data.Dataset.from_tensor_slices((x_train_full, y_train))
    # valid_data = tf.data.Dataset.from_tensor_slices((x_valid_full, y_valid))

    # x_train_full = x_train_full.to_numpy()
    # x_valid_full = x_valid_full.to_numpy()

    # y_train = y_train.to_numpy()
    # y_valid = y_valid.to_numpy()

    # x_train_full = x_train_full.reshape(
    #     1, len(x_train_full), len(x_train_full[1]))

    # x_valid_full = x_valid_full.reshape(
    #     1, len(x_valid_full), len(x_valid_full[1]))

    # print(x_train_full, '\n', x_train_full.shape, '\n')
    # print(x_valid_full, '\n', x_valid_full.shape, '\n')
    # print(y_train, '\n', y_train.shape, '\n')
    # print(y_valid, '\n', y_valid.shape, '\n')

    model = keras_model(x=x_train_full,
                        y=y_train)
    history = model.fit(x_train_full, y_train, batch_size=288,
                        epochs=10, verbose=1, callbacks=None)
