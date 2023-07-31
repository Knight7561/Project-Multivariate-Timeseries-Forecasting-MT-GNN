# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data_path', type=str, default='../data/solar.txt',
                    help='location of the data file')
parser.add_argument('--output_file_name', type=str, default='solar.pt', help='output_file_name')
parser.add_argument('--output_path', type=str, default='results/', help='location of the results')
parser.add_argument('--normalize', type=int, default=2, help='Normalise')
parser.add_argument('--epochs', type=int, default=30, help='Epochs')
parser.add_argument('--batch_size', type=int, default=30, help='Batch Size')
parser.add_argument('--sample_data', type=int, default=0, help='Sample data')
parser.add_argument('--num_nodes', type=int, default=6, help='Number of nodes')

args = parser.parse_args()

data = pd.read_csv(args.data_path, header=None)

data = np.array(data)


class DataUtil(object):
    def __init__(self, filename, train=0.6, valid=0.2, horizon=12, window=24 * 7, normalise=2, sample_data=None):
        try:

            self.rawdata = np.loadtxt(open(filename), delimiter=',')

            self.w = window
            self.h = horizon
            self.data = np.zeros(self.rawdata.shape)
            self.n, self.m = self.data.shape
            self.normalise = normalise
            self.scale = np.ones(self.m)

            self.normalise_data(normalise)

            # Run on sample data
            if sample_data:
                self.data = self.data[:sample_data]
                self.n, self.m = self.data.shape

            self.split_data(train, valid)
        except IOError as err:
            print("Error opening data file ... %s", err)

    def normalise_data(self, normalise):
        print("Normalise: %d", normalise)

        if normalise == 0:  # do not normalise
            self.data = self.rawdata

        if normalise == 1:  # same normalisation for all timeseries
            self.data = self.rawdata / np.max(self.rawdata)

        if normalise == 2:  # normalise each timeseries alone. This is the default mode
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdata[:, i]))
                self.data[:, i] = self.rawdata[:, i] / self.scale[i]

    def split_data(self, train, valid):
        print("Splitting data into training set {}, validation set {} and testing set {}".format(train, valid,
                                                                                                 1 - (train + valid)))

        train_set = range(self.w + self.h - 1, int(train * self.n))
        valid_set = range(int(train * self.n), int((train + valid) * self.n))
        test_set = range(int((train + valid) * self.n), self.n)

        self.train = self.get_data(train_set)
        self.valid = self.get_data(valid_set)
        self.test = self.get_data(test_set)

    def get_data(self, rng):
        n = len(rng)

        X = np.zeros((n, self.w, self.m))
        Y = np.zeros((n, self.w, self.m))

        for i in range(n):
            end = rng[i] - self.h + 1
            start = end - self.w

            X[i, :, :] = self.data[start:end, :]
            Y[i, :, :] = self.data[start + 1:end + 1, :]

        return [X, Y]


window = 24 * 7

Data = DataUtil(args.data_path, window=window, normalise=args.normalize, sample_data=args.sample_data)

baseline = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window, args.num_nodes)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(args.num_nodes)
])
baseline.summary()


def rse(y_true, y_pred):
    num = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=None))
    den = tf.keras.backend.std(y_true, axis=None)
    return tf.math.divide_no_nan(num, den)


def corr(y_true, y_pred):
    num1 = y_true - tf.keras.backend.mean(y_true, axis=0)
    num2 = y_pred - tf.keras.backend.mean(y_pred, axis=0)

    num = tf.keras.backend.mean(num1 * num2, axis=0)
    den = tf.keras.backend.std(y_true, axis=0) * tf.keras.backend.std(y_pred, axis=0)

    return tf.keras.backend.mean(tf.math.divide_no_nan(num, den))


baseline.compile(loss='mae', optimizer='adam', metrics=[rse, corr])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)

baseline.fit(Data.train[0], Data.train[1], validation_data=(Data.valid[0], Data.valid[1]),
             epochs=args.epochs, batch_size=args.batch_size)
Path(args.output_path).mkdir(exist_ok=True)
baseline.save(args.output_path + args.output_file_name)

test_labels = Data.test[1]
labels = []
for i in range(test_labels.shape[0]):
    labels.append(test_labels[i, -1, :])
labels = np.vstack(labels)

test_predictions = baseline.predict(Data.test[0])

predictions = []
for i in range(test_predictions.shape[0]):
    predictions.append(test_predictions[i, -1, :])
predictions = np.vstack(predictions)

scaled_predictions = np.zeros(predictions.shape)
for i in range(predictions.shape[1]):
    scaled_predictions[:, i] = predictions[:, i] * Data.scale[i]

np.save(f"{args.output_path}/actual_ouput", labels)
np.save(f"{args.output_path}/predicted_output", scaled_predictions)
