"""
This is module with classes and helper functions that was used for neural network model training.
"""

from time import time
from pathlib import Path
import os
import json


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import keras
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error

from sklearn.impute import SimpleImputer
import sklearn as sk
from sklearn import svm


def series_to_supervised(df, predict_columns=None, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.

    Modified from:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    """

    if predict_columns is None:
        predict_columns = list(df.columns)

    n_vars = 1 if type(df) is list else df.shape[1]

    cols, names = list(), list()
    col_names = df.columns

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{col_names[j]}(t-{i})') for j in range(n_vars)]
    total_features = len(names)

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[predict_columns].shift(-i))
        if i == 0:
            # names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            names += [(f'{cname}(t)') for cname in predict_columns]
        else:
            names += [(f'{cname}(t+{i})') for cname in predict_columns]

    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values.
    # This will drop rows that don't have enough history to make a full supervised learning row.

    if dropnan:
        agg.dropna(inplace=True)
    return agg.iloc[:, 0:total_features], agg.iloc[:, total_features:]


def json_out(d, filename):
    """ Export a dictionary to a JSON """

    with open(f"{filename}.json", "w") as save_file:
        json.dump(d, save_file)


def json_to_dict(filename):
    """ Convert a JSON to a dictionary """

    with open(f"{filename}", 'r') as file:
        d = json.load(file)
    return d


class SaveStatsCallBack(keras.callbacks.Callback):
    """ This is a Keras Callback that saves information at the end of the epoch. """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        stats = self.model.stats[self.model.name]
        stats['epochs'] += 1
        for key in keys:
            if key not in stats:
                stats[key] = []
            stats[key].append(logs[key])


class TimingCallback(keras.callbacks.Callback):
    """ A Keras Callback that saves the amount of time the epoch took to run. """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs=None):
        stats = self.model.stats[self.model.name]
        stats['runtime'].append(time() - self.starttime)


class ModelSeq(Sequential):
    """ This class saves checkpoint data.
    It also allows the resumption of training the model from a specific point. """

    def __init__(self
                 , model_layers
                 , X_train
                 , y_train
                 , X_val
                 , y_val
                 , name='model'
                 , initial_lr=.001
                 , chart_title=None
                 , reload=None
                 , notebook_name='seq_model'):
        super().__init__()

        self._name = name
        self.save_predictions = False
        self.notebook_name = notebook_name
        # Create a path for the checkpoint models
        self.checkpoints_path = f"{os.getcwd()}/model_checkpoints/{self.notebook_name}/{self.name}"
        Path(self.checkpoints_path).mkdir(parents=True, exist_ok=True)
        tf.random.set_seed(846)
        self.add_layers(model_layers)
        self.stats = {self.name: {'epochs': 0, 'runtime': []}}
        self.predictions = {self.name: {'train': [], 'val': []}}
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.metrics1 = ['MSE', 'MAE', 'MAPE']
        self.lr = initial_lr
        self.call_backs = [SaveStatsCallBack(self)
            , tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoints_path + '/{epoch:02d}.hdf5',
                verbose=0,
                save_weights_only=True)
            , TimingCallback(self)
                           ]

        if chart_title is None:
            self.chart_title = self.name
        else:
            self.chart_title = chart_title
        self.compile_model(self.lr)

        if reload is not None:
            self.reload_model(reload)
            print(f"This model [{self.name}] was reloaded from Epoch {reload}")
            print(f"Loss: {round(self.stats[self.name]['loss'][reload - 1], 3)}")
            print(f"Validation Loss: {round(self.stats[self.name]['val_loss'][reload - 1], 3)}")

    def add_layers(self, layers1):
        """ Adds the layers to the model. """

        for layer in layers1:
            self.add(layer)

    def compile_model(self, learning_rate=None, **kwargs):
        """ Compiles the model with saved information. A new learning rate can be passed. """

        if learning_rate is None:
            learning_rate = self.lr
        self.compile(loss=self.loss_fn,
                     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     metrics=self.metrics1, **kwargs)

    def train_model(self, epochs, batch_size=64, save_predictions=False, **kwargs):
        """Calls the Keras function to train the fit the model. Also saves out the stats to a dictionary."""

        if save_predictions:
            self.save_predictions = True

        self.fit(
            x=self.X_train,
            y=self.y_train,
            validation_data=(self.X_val, self.y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.call_backs,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=self.stats[self.name]['epochs'],
            **kwargs
        )

        # Write out a dictionary with the stats after the model is trained
        json_out(self.stats, f"{self.checkpoints_path}/{self.name}_dict")

        # Reset the flag
        self.save_predictions = False

    def rename(self, new_name):
        self.stats[new_name] = self.stats.pop(self.name)
        self.checkpoints_path = f"{os.getcwd()}/model_checkpoints/{self.notebook_name}/{new_name}"
        self._name = new_name

    def restore_dict(self, d, check_point_number):
        """Restore the stats to the model up to the checkpoint"""

        dict_name = self.name
        tmp_d = {}
        for stat in d[dict_name]:
            if stat == 'epochs':
                tmp_d[stat] = check_point_number
            else:
                tmp_d[stat] = d[dict_name][stat][:check_point_number]
        return {dict_name: tmp_d}

    def reload_model(self, checkpoint_number):
        """Reload previous weights and stats"""

        self.stats = self.restore_dict(json_to_dict(f"{self.checkpoints_path}/{self.name}_dict.json"),
                                       checkpoint_number)
        self.load_weights(f"{self.checkpoints_path}/{str(checkpoint_number).zfill(2)}.hdf5")


def shape_for_lstm(X):
    """Reshape the data for a LSTM model"""

    X_2 = X.reshape((X.shape[0], 1, X.shape[1]))
    return X_2


def shape_for_CNN(X, n_features, n_steps=672):
    """Rehshapes the data for a CNN model."""
    X_2 = X.reshape(X.shape[0], n_steps, n_features)
    return X_2


def plot_models_together(d, filename=None, no_title=False):
    """Plots all the metric curves on 4 plots in 1 figure"""

    fig, axes = plt.subplots(3, 2)
    # fig_sing, axes_sing = plt.subplots(1.1)
    fig.set_figheight(18)
    fig.set_figwidth(10)

    # Allow an argument to pass metrics
    plot_types = ['MSE', 'val_MSE', 'MAPE', 'val_MAPE', 'MAE', 'val_MAE']

    for p in plot_types:
        if p == 'val_MSE':
            ylab = 'MSE'
            title = 'Validation MSE'
            lim = [0, 1.5]
            ax = axes[0, 0]
        elif p == 'MSE':
            ylab = 'MSE'
            title = 'Training MSE'
            lim = [0, 1.5]
            ax = axes[0, 1]
        elif p == 'val_MAPE':
            ylab = 'MAPE'
            title = 'Validation MAPE'
            lim = [0, 150]
            ax = axes[1, 0]
        elif p == 'MAPE':
            ylab = 'MAPE'
            title = 'Training MAPE'
            lim = [0, 150]
            ax = axes[1, 1]
        elif p == 'val_MAE':
            ylab = 'MAE'
            title = 'Validation MAE'
            lim = [0, 1.5]
            ax = axes[2, 0]
        elif p == 'MAE':
            ylab = 'MAE'
            title = 'Training MAE'
            lim = [0, 1.5]
            ax = axes[2, 1]
        else:
            sys.exit('Plot type not defined')

        # Add data from all the models
        for model in d:
            # print(model)
            train_acc = (d[model][p])
            x_epochs = np.arange(1, len(d[model][p]) + 1)
            ax.plot(x_epochs, train_acc, label=model)

        ax.set_ylim(lim)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylab)

        ax.set_title(title, fontsize=12)
        ax.legend()

    if not no_title:
        plt.suptitle('Model Comparison', fontsize=14, y=.91)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def plot_prediction(y, y_pred, range_num=480
                    , start_num=9600, figure_size=(3, 3), filename=None
                    , title=None, legend=False):
    """Plot out a segment of the prediction """
    fig, ax = plt.subplots()

    ax.plot(np.arange(start_num, range_num + start_num), y[start_num:range_num + start_num], label='Actual')
    ax.plot(np.arange(start_num, range_num + start_num), y_pred[start_num:range_num + start_num], label='Predicted')
    plt.title(title, fontsize=14)
    if legend:
        plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
    # return ax


def plot_prediction_mulitple(y, y_pred, model_title, range_num=480
                             , start_num=9600, figure_size=(10, 18), filename=None
                             , title='Prediction vs Actual', legend=False):
    """Plot out a segment of the prediction """
    fig, ax = plt.subplots(4, 2)
    x_values = np.arange(start_num, range_num + start_num)
    fig.set_figwidth(figure_size[0])
    fig.set_figheight(figure_size[1])

    plot_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    for i, sub_plot in enumerate(plot_list):
        ax[sub_plot].plot(x_values, y[i][start_num:range_num + start_num], label='Actual')
        ax[sub_plot].plot(x_values, y_pred[i][start_num:range_num + start_num], label='Predicted')
        ax[sub_plot].set_xlabel('Time')
        ax[sub_plot].set_ylabel('kwh')
        ax[sub_plot].set_title(model_title[i], fontsize=12)
        if legend:
            ax[sub_plot].legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.suptitle(title, fontsize=14, y=.91)
    plt.show()
    # return ax


def create_summary_table(d):
    """ Creates a summary table from a dictionary """

    min_mse = []
    min_mape = []
    min_mae = []
    min_val_mse = []
    min_val_mae = []
    epoch_min_val_mape = []
    epoch_min_val_mse = []
    epoch_min_val_mae = []
    min_val_mape = []
    row_names = []
    runtime = []
    for each in d:
        min_mse.append(min(d[each]['MSE']))
        min_mape.append(min(d[each]['MAPE']))
        min_mae.append(min(d[each]['MAE']))
        min_val_mse.append(min(d[each]['val_MSE']))
        min_val_mape.append(min(d[each]['val_MAPE']))
        min_val_mae.append(min(d[each]['val_MAE']))
        epoch_min_val_mape.append(np.argmin(d[each]['val_MAPE']) + 1)
        epoch_min_val_mse.append(np.argmin(d[each]['val_MSE']) + 1)
        epoch_min_val_mae.append(np.argmin(d[each]['val_MAE']) + 1)
        row_names.append(each)
    df = pd.DataFrame(list(zip(row_names
                               , min_mse
                               , min_mape
                               , min_mae
                               , min_val_mse
                               , min_val_mape
                               , min_val_mae
                               , epoch_min_val_mape
                               , epoch_min_val_mse
                               , epoch_min_val_mae)),
                      columns=[
                          'Model'
                          , 'Minium MSE'
                          , 'Minimum MAPE'
                          , 'Minimum MAE'
                          , 'Minimum Val MSE'
                          , 'Minimum Val MAPE'
                          , 'Minimum Val MAE'
                          , 'Epoch of Min Val MAPE'
                          , 'Epoch of Min Val MSE'
                          , 'Epoch of Min Val MAE']).round(3)
    # display(df)
    return df


def create_summary_table_test(d):
    """Creates a summary table from a dictionary, specifically for the using the Test dataset"""

    min_mse = []
    min_mape = []
    min_mae = []
    row_names = []
    runtime = []
    for each in d:
        min_mse.append(min(d[each]['MSE']))
        min_mape.append(min(d[each]['MAPE']))
        min_mae.append(min(d[each]['MAE']))
        row_names.append(each)
    df = pd.DataFrame(list(zip(row_names, min_mse, min_mape, min_mae)), columns=['Model', 'MSE', 'MAPE', 'MAE']).round(
        3)
    # display(df)
    return (df)
