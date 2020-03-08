import argparse
import datetime
import logging
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time


def read_args():
    """Read the command line arguments and returns the populated namespace"""
    parser = argparse.ArgumentParser(
        description='Predicting bitcoin prices with RNN and LSTM')
    parser.add_argument('csv_filename', nargs=1, help='csv file name')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='execution on google colab, mounts gdrive')
    return parser.parse_args()


def date_parser(time_in_secs):
    """Conversion function for native timestamps in the csv file"""
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def plot(data_list):
    """Plot a DataFrame using matplotlib"""
    plt.style.use('ggplot')
    plt.close('all')
    for data in data_list:
        plt.plot(data)
    plt.show()


def dataframe_info(data):
    """Build a string with useful characteristics of the dataframe"""
    return f'data: {"*".join(map(str,data.shape))} elements, ' + \
           f'{data.memory_usage().sum():_} bytes'


def mount_gdrive(path='/content/gdrive'):
    from google.colab import drive
    drive.mount(path)


def read_values(csv_filename):
    """Retrieve the csv contents and clean things up"""
    start = time.process_time()
    data = pd.read_csv(csv_filename, parse_dates=True,
                       date_parser=date_parser,
                       index_col=[0])
    log.debug('csv import took %.2f seconds', time.process_time() - start)
    log.debug(dataframe_info(data))

    cols_to_keep = ['Open', 'Volume_(BTC)']
    log.info('keeping only the following column(s): ' + ','.join(cols_to_keep))
    for col_name in data.columns:
        if col_name not in cols_to_keep:
            data.drop(col_name, axis=1, inplace=True)
    log.debug(dataframe_info(data))

    log.info('removing NaN rows')
    data.dropna(inplace=True)
    log.debug(dataframe_info(data))

    return data


def resample(data):
    resampled = data.resample('1D').first()
    resampled.dropna(inplace=True)
    log.debug(dataframe_info(resampled))
    log.debug('resampled data:\n' + resampled.to_string(max_rows=30))
    return resampled


def prepare(data):
    npdata = data.values  # convert to nparray
    training_data_len = math.ceil(len(npdata)*0.8)
    log.info('training data length: %d', training_data_len)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(npdata)

    x_train = []
    y_train = []
    for i in range(60, training_data_len):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    log.info('recreated training data length: %d', len(x_train))

    x_val = []
    y_val = []
    for i in range(training_data_len+60, len(scaled_data)):
        x_val.append(scaled_data[i-60:i, 0])
        y_val.append(scaled_data[i, 0])
    log.info('recreated validation data length: %d', len(x_val))

    return (np.array(x_train), np.array(y_train)),\
           (np.array(x_val), np.array(y_val))


def create_LSTM():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=1e-5)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.summary()

    return model


def train_LSTM(x_train,
               y_train,
               x_val,
               y_val,
               model=None,
               epochs=1,
               ckpt_basedir='ckpt',
               log_basedir='log'):
    current_datetime = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    checkpoint_path = os.path.join(ckpt_basedir, current_datetime)
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)

    # Callback for tensorboard
    log_path = os.path.join(log_basedir, 'fit', current_datetime)
    tb_callback = TensorBoard(log_dir=log_path,
                              profile_batch=0,  # tf bug 2412 workaround
                              histogram_freq=1)

    # need a 3rd column for LSTM
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_val = np.reshape(x_val, x_val.shape + (1,))

    if model is not None:
        model.fit(x_train, y_train,
                  batch_size=1,
                  epochs=epochs,
                  validation_data=(x_val, y_val),
                  callbacks=[tb_callback, cp_callback])  # order is important

    plot([y_val, model.predict(x_val)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)
    log.info('starting')
    args = read_args()

    if args.colab:
        log.info('mounting gdrive')
        mount_gdrive()

    log.info('reading csv')
    data = read_values(args.csv_filename[0])
    log.debug('sample data:\n' + data.to_string(max_rows=30))

    log.info('resampling data')
    resampled_data = resample(data)

    log.info('preparing training data')
    (x_train, y_train), (x_val, y_val) = prepare(resampled_data)

    model = create_LSTM()
    train_LSTM(x_train, y_train,
               x_val, y_val,
               model=model, epochs=5)
