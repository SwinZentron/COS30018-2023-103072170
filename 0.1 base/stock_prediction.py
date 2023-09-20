# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

from msilib import Feature
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import pickle
import csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dropout, Dense

import matplotlib.pyplot as plt

#task 5
#   multi step prediction function
def create_sequences_multistep(data, seq_length, n_steps):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-n_steps):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length):(i+seq_length+n_steps)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_sequences_multivariate(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, -1]  # assuming target column is the last one
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_sequences_combination(data, seq_length, n_steps):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-n_steps):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length):(i+seq_length+n_steps)]
        xs.append(x)
        ys.append(y[:, -1])  # assuming target column is the last one
    return np.array(xs), np.array(ys)


#task 4
def create_model(sequence_length, n_features, units=[256], cells=['LSTM'], n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    # Create a Sequential model
    model = Sequential()
    
    # Loop over the number of layers
    for i in range(n_layers):
        # Get the name of the cell (layer type) for this layer
        cell_name = cells[i]
        
        # Check if the cell name corresponds to a valid layer network type
        if cell_name not in globals():
            raise ValueError(f"Invalid layer network type: {cell_name}")
        
        # Get a reference to the corresponding layer network object
        cell = globals()[cell_name]
        
        # Get the number of units for this layer
        unit = units[i]
        
        # If this is the first layer...
        if i == 0:
            if bidirectional:
                model.add(Bidirectional(cell(unit, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(unit, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
                
        # If this is the last layer...
        elif i == n_layers - 1:
            if bidirectional:
                model.add(Bidirectional(cell(unit, return_sequences=False)))
            else:
                model.add(cell(unit, return_sequences=False))
                
        # If this is a hidden layer...
        else:
            if bidirectional:
                model.add(Bidirectional(cell(unit, return_sequences=True)))
            else:
                model.add(cell(unit, return_sequences=True))
                
        # Add dropout after each layer
        model.add(Dropout(dropout))
    
    # Add a Dense output layer with linear activation
    model.add(Dense(1, activation="linear"))
    
    # Compile the model with the specified loss function and optimizer
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    # Return the compiled model
    return model

#task 3
def plot_boxplot(df, n, columns):
    # Calculate the rolling window data for each column
    rolling_data = [df[column].rolling(n).mean() for column in columns]
    
    # Create the box plot
    fig, ax = plt.subplots()
    ax.boxplot([data.dropna() for data in rolling_data], labels=columns)
    ax.set_title(f'{n} Day Rolling Window')
    
    # Show the plot
    plt.show()

#task 3
def plot_candlestick(df, n=1):
    # Resample the data to have one row per n trading days
    df = df.resample(f'{n}D').agg({'Open': 'first', 
                                    'High': 'max', 
                                    'Low': 'min', 
                                    'Close': 'last'})
    
    # Create the candlestick chart
    mpf.plot(df, type='candle')

#task 2
def downloadData(ticker, start_date, end_date, save_file=False):
     #create data folder in working directory if it doesnt already exist
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data = None 
    #if ticker is a string, load it from yfinance library
    if isinstance(ticker, str):
        # Check if data file exists based on ticker, start_date and end_date
        file_path = os.path.join(data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        if os.path.exists(file_path):
            # Load data from file
            data = pd.read_csv(file_path)
        else:
            # Download data using yfinance
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Save data to file if boolean save_file is True
            if save_file:
                data.to_csv(file_path) 
    #if passed in ticker is a dataframe, use it directly
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        data = ticker
    else:
        # raise error if ticker is neither a string nor a dataframe
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # return the dataframe
    return data

def processNANs(df, fillna_method):
    # Deal with potential NaN values in the data
    # Drop NaN values
    if fillna_method == 'drop':
        df.dropna(inplace=True)
    #use forward fill method, fill NaN values with the previous value
    elif fillna_method == 'ffill':
        df.fillna(method='ffill', inplace=True)
    #use backward fill method, fill NaN values with the next value
    elif fillna_method == 'bfill':
        df.fillna(method='bfill', inplace=True)
    #use mean method, fill NaN values with the mean of the column
    elif fillna_method == 'mean':
        df.fillna(data.mean(), inplace=True)

    return df
#task 2 - function to load and process a dataset with multiple features
def processData(
    ticker, 
    start_date, 
    end_date, 
    save_file, 
    prediction_column, 
    prediction_days, 
    feature_columns=[], 
    split_method='date', 
    split_ratio=0.8, 
    split_date=None, 
    fillna_method='drop', 
    scale_features=False, 
    scale_min=0, 
    scale_max=1,
    save_scalers=False,
    n_steps=1,  # number of future days to predict
    task5method=2):  # whether to use multiple features
    """
    Load and process a dataset with multiple features.
    
    :param ticker: str, company ticker symbol
    :param start_date: str, start date for the dataset in the format 'YYYY-MM-DD'
    :param end_date: str, end date for the dataset in the format 'YYYY-MM-DD'
    :param save_file: bool, whether to save the dataset to a file
    :param prediction_days: int, number of days to predict into the future
    :param feature_columns: list, list of feature columns to use in the model
    :param split_method: str, method to split the data into train/test data ('date' or 'random')
    :param split_ratio: float, ratio of train/test data if split_method is 'random'
    :param split_date: str, date to split the data if split_method is 'date'
    :param fillna_method: str, method to drop or fill NaN values in the data ('drop', 'ffill', 'bfill', or 'mean')
    :param scale_features: bool, whether to scale the feature columns
    :param scale_min: int, minimum value to scale the feature columns
    :param scale_max: int, maximum value to scale the feature columns
    :param save_scalers: bool, whether to save the scalers to a file
    :param n_steps: int, number of future days to predict
    :param multivariate: bool, whether to use multiple features
    :return: tuple of pandas.DataFrame, train and test data
    """
    
    data = downloadData(ticker, start_date, end_date, save_file)
   
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = data.copy()
   
    # make sure that the passed feature_columns exist in the dataframe
    if len(feature_columns) > 0:
        for col in feature_columns:
            assert col in data.columns, f"'{col}' does not exist in the dataframe."
    else:
        # if no feature_columns are passed, use all columns except the prediction_column
        feature_columns = list(filter(lambda column: column != 'Date', data.columns))
    
    # add feature columns to result
    result['feature_columns'] = feature_columns
    # Deal with potential NaN values in the data
    # Drop NaN values
    data = processNANs(data, fillna_method)

    # Split data into train and test sets based on date
    if split_method == 'date':
        train_data = data.loc[data['Date'] < split_date]
        test_data = data.loc[data['Date'] >= split_date]
    # Split data into train and test sets randomly with provided ratio
    elif split_method == 'random':
        train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
    # Reset index of both dataframes
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    # Sort dataframes by date
    train_data = train_data.sort_values(by='Date')
    test_data = test_data.sort_values(by='Date')

    # Scale features
    if scale_features:
        # Create scaler dictionary to store all scalers for each feature column
        scaler_dict = {}
        # Dictionaries to store scaled train and test data
        scaled_train_data = {}
        scaled_test_data = {}
        #loop through each feature column
        for col in feature_columns:
            # Create scaler for each feature column using Min Max, passing in the scale_min and scale_max
            scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
            # Fit and transform scaler on train data
            scaled_train_data[col] = scaler.fit_transform(train_data[col].values.reshape(-1, 1)).ravel()
            # Transform test data using scaler
            scaled_test_data[col] = scaler.transform(test_data[col].values.reshape(-1,1)).ravel()
            # Add scaler to scaler dictionary, using the feature column name as key
            scaler_dict[col] = scaler
        # Add scaler dictionary to result
        result["column_scaler"] = scaler_dict
        
         # Save scalers to file
        if save_scalers:
            # Create scalers directory if it doesn't exist
            scalers_dir = os.path.join(os.getcwd(), 'scalers')
            if not os.path.exists(scalers_dir):
                os.makedirs(scalers_dir)
            # Create scaler file name
            scaler_file_name = f"{ticker}_{start_date}_{end_date}_scalers.txt"
            scaler_file_path = os.path.join(scalers_dir, scaler_file_name)
            with open(scaler_file_path, 'wb') as f:
                pickle.dump(scaler_dict, f)
       
        # Convert scaled data to dataframes
        train_data = pd.DataFrame(scaled_train_data)
        test_data = pd.DataFrame(scaled_test_data)

    # Add train and test data to result
    result["scaled_train"] = train_data
    result["scaled_test"] = test_data
    # Construct the X's and y's for the training data
    X_train, y_train = [], []

    #Task 5
    # Create sequences
    if task5method == 1: #only multistep
        X_train, y_train = create_sequences_multistep(train_data[feature_columns], seq_length=prediction_days, n_steps=n_steps)
        X_test, y_test = create_sequences_multistep(test_data[feature_columns], seq_length=prediction_days, n_steps=n_steps)
    elif task5method == 2: #only multivariate
        X_train, y_train = create_sequences_multivariate(train_data[prediction_column], seq_length=prediction_days)
        X_test, y_test = create_sequences_multivariate(test_data[prediction_column], seq_length=prediction_days)
    elif task5method == 3: #both
        X_train, y_train = create_sequences_combination(train_data[prediction_column], seq_length=prediction_days, n_steps=n_steps)
        X_test, y_test = create_sequences_combination(test_data[prediction_column], seq_length=prediction_days, n_steps=n_steps)
    
    """
    # Loop through the training data from prediction_days to the end
    for x in range(prediction_days, len(train_data)):
        # Append the values of the passed prediction column to X_train and y_train
        X_train.append(train_data[prediction_column].iloc[x-prediction_days:x])
        y_train.append(train_data[prediction_column].iloc[x])
    """
    # convert to numpy arrays
    result["X_train"] = np.array(X_train)
    result["y_train"] = np.array(y_train)
    # reshape X_train for proper fitting into LSTM model
    result["X_train"] = np.reshape(result["X_train"], (result["X_train"].shape[0], result['X_train'].shape[1], -1));

    """
    # construct the X's and y's for the test data
    X_test, y_test = [], []
    # Loop through the test data from prediction_days to the end
    for x in range(prediction_days, len(test_data)):
        # Append the values of the passed prediction column to X_test and y_test
        X_test.append(test_data[prediction_column].iloc[x - prediction_days:x])
        y_test.append(test_data[prediction_column].iloc[x])
    """
    # convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #assign y_test to result
    result["y_test"] = y_test
    #assign X_test to result and reshape X_test for prediction compatibility
    result["X_test"] = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1));

    return result

#------------------------------------------------------------------------------
# Load and process data using Task B.2 Function
# ------------------------------------------------------------------------------

# define function parameters to use
DATA_SOURCE = "yahoo"
COMPANY = "TSLA"  
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2022-12-31'
SAVE_FILE = True
PREDICTION_DAYS = 100
SPLIT_METHOD = 'random'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2020-01-02'
NAN_METHOD = 'drop'
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
SCALE_FEATURES = True
SCALE_MIN = 0
SCALE_MAX = 1
SAVE_SCALERS = True
prediction_column = "Close"

# Call processData function passing in parameters
data = processData(
    ticker=COMPANY, 
    start_date=DATA_START_DATE, 
    end_date=DATA_END_DATE, 
    save_file=SAVE_FILE,
    prediction_column=prediction_column,
    prediction_days=PREDICTION_DAYS,
    split_method=SPLIT_METHOD, 
    split_ratio=SPLIT_RATIO, 
    split_date=SPLIT_DATE,
    fillna_method=NAN_METHOD,
    feature_columns=FEATURE_COLUMNS,
    scale_features=SCALE_FEATURES,
    scale_min=SCALE_MIN,
    scale_max=SCALE_MAX,
    save_scalers=SAVE_SCALERS
    )

plot_candlestick(processNANs(downloadData(COMPANY, '2022-05-01', '2022-05-31', False), 'drop'), 5)

#plot_boxplot(processNANs(downloadData(COMPANY, '2019-01-01', '2022-12-31', False),'drop'),['Open', 'High', 'Low', 'Close', 'Adj Close'], 10)

plot_boxplot(downloadData(COMPANY, '2019-01-01', '2022-12-31', False), 40, ['Open', 'High', 'Low', 'Close', 'Adj Close'])
# Number of days to look back to base the prediction
#PREDICTION_DAYS = 60 # Original

#------------------------------------------------------------------------------
#Task 4
sequence_length = data['X_train'].shape[1]
n_features = data['X_train'].shape[2]
#set 1

units = [256, 128]
cells = ['LSTM', 'GRU']
n_layers = 2
dropout = 0.3
loss = "mean_absolute_error"
optimizer = "rmsprop"
bidirectional = True

# Set the number of epochs and batch size
epochs = 25
batch_size = 32


#set 2
"""
# Set the model parameters
units = [256, 128, 64]
cells = ['LSTM', 'GRU', 'SimpleRNN']
n_layers = 3
dropout = 0.2
loss = "mean_squared_error"
optimizer = "adam"
bidirectional = True

# Set the training parameters
epochs = 25
batch_size = 32
"""
#set 3
"""
# Set the model parameters
units = [512, 256]
cells = ['GRU', 'GRU']
n_layers = 2
dropout = 0.4
loss = "mean_absolute_percentage_error"
optimizer = "sgd"
bidirectional = False

# Set the training parameters
epochs = 35
batch_size = 64
"""

#set 4
"""
# Set the model parameters
units = [128, 64, 32]
cells = ['SimpleRNN', 'SimpleRNN', 'SimpleRNN']
n_layers = 3
dropout = 0.5
loss = "huber_loss"
optimizer = "adagrad"
bidirectional = True

# Set the training parameters
epochs = 15
batch_size = 16
"""


# Create the model using the create_model function
model = create_model(sequence_length, n_features, units=units, cells=cells, n_layers=n_layers,
                     dropout=dropout, loss=loss, optimizer=optimizer, bidirectional=bidirectional)



# Train the model on the training data
model.fit(data['X_train'], data['y_train'], epochs=epochs, batch_size=batch_size)



actual_prices = data["column_scaler"][prediction_column].inverse_transform(data["y_test"].reshape(-1,1))

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------

predicted_prices = model.predict(data['X_test'])
#print(predicted_prices.shape)
predicted_prices = data["column_scaler"][prediction_column].inverse_transform(predicted_prices)
#predicted_prices = predicted_prices.ravel()
#predicted_prices = np.concatenate((np.full(PREDICTION_DAYS, np.nan), predicted_prices))

# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

predicted_prices = predicted_prices.ravel()
actual_prices = actual_prices.ravel()
df = pd.DataFrame(predicted_prices)
df.to_csv('predicted_prices.csv', index=False)
df = pd.DataFrame(actual_prices)
df.to_csv('actual_prices.csv', index=False)
#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [data['X_test'][len(data['X_test']) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = data["column_scaler"][prediction_column].inverse_transform(prediction)
print(f"Prediction: {prediction[0]}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??