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
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

#task 2 - function to load and process a dataset with multiple features
def processData(ticker, start_date, end_date, save_file, prediction_days, feature_columns=[], split_method='date', split_ratio=0.8, split_date=None, fillna_method='drop', scale_features=False, scale_min=0, scale_max=1 , save_scalers=False):
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
    
    :return: tuple of pandas.DataFrame, train and test data
    """
    
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
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
   
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = data.copy()
   
    # make sure that the passed feature_columns exist in the dataframe
    if len(feature_columns) > 0:
        for col in feature_columns:
            assert col in data.columns, f"'{col}' does not exist in the dataframe."
    else:
        feature_columns = list(filter(lambda column: column != 'Date', data.columns))
    
    result['feature_columns'] = feature_columns
    # Deal with potential NaN values in the data
    # Drop NaN values
    if fillna_method == 'drop':
       data.dropna(inplace=True)
       print("drop")
    #use forward fill method, fill NaN values with the previous value
    elif fillna_method == 'ffill':
        data.fillna(method='ffill', inplace=True)
    #use backward fill method, fill NaN values with the next value
    elif fillna_method == 'bfill':
        data.fillna(method='bfill', inplace=True)
    #use mean method, fill NaN values with the mean of the column
    elif fillna_method == 'mean':
        data.fillna(data.mean(), inplace=True)
 
    #data['Date'] = pd.to_datetime(data['Date'])
    #split_date = pd.to_datetime(split_date)
    #train_mask = data['Date'] < split_date
    #test_mask = data['Date'] >= split_date

    # Split data into train and test sets
    if split_method == 'date':
        train_data = data.loc[data['Date'] < split_date]
        test_data = data.loc[data['Date'] >= split_date]

        #train_data = data[data['Date'] < split_date]
        #test_data = data[data['Date'] >= split_date]
    elif split_method == 'random':
        train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
   
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_data = train_data.sort_values(by='Date')
    test_data = test_data.sort_values(by='Date')
    

    #train_data = train_data.reshape(-1, 1)
    #test_data = test_data.reshape(-1, 1)

    # Scale features
    if scale_features:
        scaler_dict = {}
        scaled_train_data = {}
        scaled_test_data = {}
        for col in feature_columns:
            scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
            scaled_train_data[col] = scaler.fit_transform(train_data[col].values.reshape(-1, 1)).ravel()
            scaled_test_data[col] = scaler.transform(test_data[col].values.reshape(-1,1)).ravel()
            scaler_dict[col] = scaler
        result["column_scaler"] = scaler_dict
        
         # Save scalers to file
        if save_scalers:
            # Create scalers directory if it doesn't exist
            scalers_dir = os.path.join(os.getcwd(), 'scalers')
            if not os.path.exists(scalers_dir):
                os.makedirs(scalers_dir)
        
            scaler_file_name = f"{ticker}_{start_date}_{end_date}_scalers.txt"
            scaler_file_path = os.path.join(scalers_dir, scaler_file_name)
            with open(scaler_file_path, 'wb') as f:
                pickle.dump(scaler_dict, f)
        with open('GFG.csv', 'w') as f:
     
        # using csv.writer method from CSV package
            write = csv.writer(f)
     
            write.writerow(feature_columns)
            write.writerows(scaled_test_data)
        
        #np.savetxt("np1.csv", np.concatenate(scaled_test_data), delimiter=",")
        #pd.DataFrame(np.concatenate(scaled_test_data)).to_csv('df1.csv', index=False)

        train_data = pd.DataFrame(scaled_train_data)
        test_data = pd.DataFrame(scaled_test_data)
        test_data.to_csv('dftest1', index=False)
        #train_data = pd.DataFrame({col: scaled_train_data[col].ravel() for col in feature_columns})
        #test_data = pd.DataFrame({col: scaled_test_data[col].ravel() for col in feature_columns})

        # Concatenate scaled data and original 'Date' column
        #train_data = pd.concat([train_data['Date'], pd.DataFrame(np.concatenate(scaled_train_data, axis=-1), columns=feature_columns)], axis=1)
        #test_data = pd.concat([test_data['Date'], pd.DataFrame(np.concatenate(scaled_test_data, axis=-1), columns=feature_columns)], axis=1)
        #result['scaled_data'] = data

    
    

    result["scaled_train"] = train_data
    result["scaled_test"] = test_data
    test_data.to_csv('df1.csv', index=False)

    
    # construct the X's and y's for the training data
    X_train, y_train = [], []

    #X = train_data.iloc[:, :-1]
    #y = train_data.iloc[:, -1]

    #scaled_train = scaled_train[:,0]
    #scaled_test = scaled_test[1:]

    #for i in range(prediction_days, len(train_data)):
    #    X_train.append(X.iloc[i-prediction_days:i].values)
    #    y_train.append(y.iloc[i])

    for x in range(prediction_days, len(train_data)):
        X_train.append(train_data['Close'].iloc[x-prediction_days:x])
        y_train.append(train_data['Close'].iloc[x])

    # convert to numpy arrays
    result["X_train"] = np.array(X_train)
    result["y_train"] = np.array(y_train)

    result["X_train"] = np.reshape(result["X_train"], (result["X_train"].shape[0], result['X_train'].shape[1], -1));
    # construct the X's and y's for the test data
    X_test, y_test = [], []

    #X = test_data.iloc[:, :-1]
    #y = test_data.iloc[:, -1]

    result["actual_values"] = test_data

    #for i in range(prediction_days, len(test_data)):
    #   X_test.append(X.iloc[i-prediction_days:i].values)
    #   y_test.append(y.iloc[i])

    for x in range(prediction_days, len(test_data)):
        X_test.append(test_data['Close'].iloc[x - prediction_days:x])
        #y_test.append(test_data['Close'].iloc[x])

    # convert to numpy arrays
    X_test = np.array(X_test)
    #result["y_test"] = np.array(y_test)

    result["X_test"] = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1));

    return result
#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
DATA_SOURCE = "yahoo"
COMPANY = "TSLA"    

# start = '2012-01-01', end='2017-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'

##data =  yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)
# yf.download(COMPANY, start = TRAIN_START, end=TRAIN_END)

# For more details: 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
PRICE_VALUE = "Close"

#scaler = MinMaxScaler(feature_range=(0, 1)) 
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
##scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 
# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape so as to maintain the same number of elements.

#------------------------------------------------------------------------------
# Load and process data using Task B.2 Function
# ------------------------------------------------------------------------------

# define function parameters to use
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2022-12-31'
SAVE_FILE = True
PREDICTION_DAYS = 60
SPLIT_METHOD = 'random'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2020-01-02'
NAN_METHOD = 'drop'
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
SCALE_FEATURES = True
SCALE_MIN = 0
SCALE_MAX = 1
SAVE_SCALERS = True

# Call processData function passing in parameters
data = processData(
    ticker=COMPANY, 
    start_date=DATA_START_DATE, 
    end_date=DATA_END_DATE, 
    save_file=SAVE_FILE,
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

# select the column to use for prediction
prediction_column = "Close"

# create X_train and y_train using only the selected column
X_train, y_train = [], []
"""
for x in range(PREDICTION_DAYS, len(data['scaled_train'])):
    X_train.append(data['scaled_train'][x-PREDICTION_DAYS:x, data['feature_columns'].index(prediction_column)])
    y_train.append(data['scaled_train'][x, data['feature_columns'].index(prediction_column)])
X_train = np.array(X_train)
y_train = np.array(y_train)
"""
# loop over the training data and create the training sets
for i in range(PREDICTION_DAYS, len(data['scaled_train'])):
    X_train.append(data['scaled_train']['Close'].iloc[i-PREDICTION_DAYS:i].values)
    y_train.append(data['scaled_train']['Close'].iloc[i])

X_train = np.array(X_train);
y_train = np.array(y_train);
# create X_test and y_test using only the selected column
X_test, y_test = [], []
"""
for x in range(PREDICTION_DAYS, len(data['scaled_test'])):
    X_test.append(data['scaled_test'][x-PREDICTION_DAYS:x, data['feature_columns'].index(prediction_column)])
    y_test.append(data['scaled_test'][x, data['feature_columns'].index(prediction_column)])

"""
for i in range(PREDICTION_DAYS, len(data['scaled_test'])):
    X_test.append(data['scaled_test']['Close'].iloc[i-PREDICTION_DAYS:i].values)
    y_test.append(data['scaled_test']['Close'].iloc[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Number of days to look back to base the prediction
#PREDICTION_DAYS = 60 # Original

"""
# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(train_data)):
    # Use the past PREDICTION_DAYS of prices as input features
    x_train.append(train_data[PRICE_VALUE].iloc[x-PREDICTION_DAYS:x])
    # Use the price of the next day as the target variable
    y_train.append(train_data[PRICE_VALUE].iloc[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 
"""


#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

#model.add(LSTM(units=50, return_sequences=True, input_shape=(data["X_train"].shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True, input_shape=(data["X_train"].shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(data['X_train'], data["y_train"], epochs=25, batch_size=32)

# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2020-01-02'
TEST_END = '2022-12-31'

#test_data = yf.download(COMPANY, start=TRAIN_START, end=TRAIN_END, progress=False)

# The above bug is the reason for the following line of code
#test_data = test_data[1:]

# Now using test_data from processData function
data["actual_values"] = data["actual_values"][1:]
actual_prices = data["column_scaler"][prediction_column].inverse_transform(data["actual_values"][prediction_column].values.reshape(-1, 1)).ravel()

#actual_prices =  data["actual_values"][prediction_column].values

#actual_prices = data["column_scaler"][prediction_column].inverse_transform(actual_prices)
# create dataframes from the y_train and y_test arrays
#y_train_df = pd.DataFrame(data['y_train'])
#y_test_df = pd.DataFrame(data['y_test'])

# concatenate the dataframes
#total_dataset = pd.concat([y_train_df, y_test_df], axis=0)

# reset the index
#total_dataset.reset_index(drop=True, inplace=True)
#total_dataset = pd.concat((data["y_train"], data["y_test"]), axis=0)
#total_dataset = data['df']

#model_inputs = total_dataset[len(total_dataset) - len(data['y_test']) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the first
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

#model_inputs = model_inputs.reshape(-1, 1)
#model_inputs = data["column_scaler"].transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------

predicted_prices = model.predict(data['X_test'])
#print(predicted_prices.shape)
predicted_prices = data["column_scaler"][prediction_column].inverse_transform(predicted_prices)
predicted_prices = np.concatenate((np.full(PREDICTION_DAYS, np.nan), predicted_prices))

"""
# get the column_scaler dictionary
column_scaler = data['column_scaler']

# create a dictionary to store the inverse transformed data for each column
inverse_transformed_data = {}

# iterate over each column and apply the inverse transform
for col, scaler in column_scaler.items():
    # get the index of the column in the predicted data
    col_idx = list(column_scaler.keys()).index(col)
    
    # select the data for the current column
    col_data = predicted_prices[:, col_idx]
    #col_data = predicted_prices[col]
    # reshape the data to have shape (n_samples, 1)
    col_data = col_data.reshape(-1, 1)
    
    # apply the inverse transform to the data
    inverse_transformed_data[col] = scaler.inverse_transform(col_data)

# concatenate the inverse transformed data for each column along axis 1
predicted_prices = np.concatenate(list(inverse_transformed_data.values()), axis=1)
"""
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

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [X_test[len(X_test) - PREDICTION_DAYS:, 0]]
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