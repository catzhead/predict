import argparse
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
import logging
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time


LIMIT_TRAINING = False


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


def plot(data):
    """Plot a DataFrame using matplotlib"""
    plt.style.use('ggplot')
    plt.close('all')
    data.plot()
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


def prepare(data):
    npdata = data.values
    training_data_len = math.ceil(len(npdata)*0.8)
    if LIMIT_TRAINING:
        training_data_len = 10000
    log.info('training data length: %d', training_data_len)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(npdata)

    x_train = []
    y_train = []
    for i in range(60, training_data_len):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    return np.array(x_train), np.array(y_train)


def create_LSTM():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()

    return model


def train_LSTM(x_train, y_train, model=None, ckpt_basedir='ckpt'):
    current_datetime = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    checkpoint_path = os.path.join(ckpt_basedir, current_datetime)
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)

    # need a 3rd column for LSTM
    x_train = np.reshape(x_train, x_train.shape + (1,))

    model.fit(x_train, y_train, batch_size=1, epochs=1,
              callbacks=[cp_callback])


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
    log.debug('Sample data:\n' + data.to_string(max_rows=10))

    x_train, y_train = prepare(data)
    model = create_LSTM()
    train_LSTM(x_train, y_train, model=model)
