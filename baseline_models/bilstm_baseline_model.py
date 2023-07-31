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
parser.add_argument('--print_batch', type=int, default=100, help='Log after how many batches')

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
            print("data shape is ", self.data.shape)

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

inputs = tf.keras.layers.Input(shape=(window, args.num_nodes))
bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
dropout = tf.keras.layers.Dropout(0.2)(bilstm1)
bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(dropout)
dropout2 = tf.keras.layers.Dropout(0.2)(bilstm2)
output = tf.keras.layers.Dense(args.num_nodes)(dropout2)

baseline_bilstm_model = tf.keras.models.Model(inputs, outputs=output)
baseline_bilstm_model.summary()


def rse(y_true, y_pred):
    num = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=None))
    den = tf.keras.backend.std(y_true, axis=None)
    return num / den


def corr_func(actual_label, predicted_label):
    sigma_p = (predicted_label).std(axis=0)
    sigma_g = (actual_label).std(axis=0)
    mean_p = predicted_label.mean(axis=0)
    mean_g = actual_label.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predicted_label - mean_p) * (actual_label - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    return correlation


epochs = args.epochs
batch_size = args.batch_size


def evaluate_model(data, labels):
    epoch_loss, epoch_rse, epoch_corr = [], [], []
    for index in range(0, len(data), batch_size):
        batch_data = data[index: index + batch_size]
        batch_labels = labels[index: index + batch_size]

        loss, rse = baseline_bilstm_model.evaluate(batch_data, batch_labels)
        predicted_labels = baseline_bilstm_model.predict(batch_data)
        corr = corr_func(batch_labels, predicted_labels)
        epoch_loss.append(loss)
        epoch_rse.append(rse)
        epoch_corr.append(corr)
    return np.mean(epoch_loss), np.mean(epoch_rse), np.mean(epoch_corr)


baseline_bilstm_model.compile(loss='mae', optimizer='adam', metrics=[rse])

train_data, train_labels = Data.train[0], Data.train[1]
valid_data, valid_labels = Data.valid[0], Data.valid[1]
test_data, test_labels = Data.test[0], Data.test[1]

for epoch in range(epochs):
    epoch_loss, epoch_rse, epoch_corr = [], [], []
    batch = 0
    for index in range(0, len(train_data), batch_size):
        batch_data = train_data[index: index + batch_size]
        batch_labels = train_labels[index: index + batch_size]

        loss, rse = baseline_bilstm_model.train_on_batch(batch_data, batch_labels)
        predicted_labels = baseline_bilstm_model.predict(batch_data)
        corr = corr_func(batch_labels, predicted_labels)
        epoch_loss.append(loss)
        epoch_rse.append(rse)
        epoch_corr.append(corr)
        if batch % args.print_batch == 0:
            print(f"Batch : {batch}, Loss : {loss}, RSE : {rse}, Correlation: {corr}")
        batch += 1
    valid_loss, valid_rse, valid_corr = evaluate_model(valid_data, valid_labels)

    print(f"Epoch : {epoch}, Loss : {np.mean(epoch_loss)}, RSE : {np.mean(epoch_rse)},"
          f" Correlation: {np.mean(epoch_corr)}, Valid Loss : {valid_loss}, Valid RSE : {valid_rse},"
          f"Valid Correlation: {valid_corr}")

test_loss, test_rse, test_corr = evaluate_model(test_data, test_labels)

print(f"Test Loss : {test_loss}, Test RSE : {test_rse}, Test Correlation: {test_corr}")

Path(args.output_path).mkdir(exist_ok=True)
baseline_bilstm_model.save(args.output_path + args.output_file_name)
