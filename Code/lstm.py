from pandas_datareader import data
from google.colab import drive
import matplotlib.pyplot as plt
import pandas as pd
import os
from numpy import concatenate, square, sum
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt, ceil, floor
from keras import Sequential
from keras.regularizers import l1,l2,l1_l2
from keras.optimizers import Adam, SGD, Adadelta,RMSprop, Nadam
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
from keras.layers import Activation, Dense, LSTM, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor

drive.mount('/content/drive')
%cd drive/'My Drive'/'Colab Notebooks'


# #load dataset and preliminary filters
df = pd.read_csv("dataraw.csv")

df.dropna(axis=1,thresh = len(df)*0.75, inplace= True)

df.dropna(axis=0,thresh = df.columns.size*0.75,inplace = True)

df.rename(index=str, inplace= True, columns={"Global Company Key - Company":"compkey", 'Historical CRSP PERMNO Link to COMPUSTAT Record': 'crspno', 'Data Year - Fiscal':'year'})

df = (df.fillna(method='ffill') + df.fillna(method='bfill'))/2

df.dropna(axis=1,inplace= True)

# #for merging drop the day component of the date
df['Date of Observation'] = df['Date of Observation'].apply(lambda x: int(str(x)[:6]))

# #drop observation which dont have the max # observations
df['num_obs'] = df.groupby('compkey')['compkey'].transform('count')
df = df[df['num_obs'] == df['num_obs'].max()]

# #drop constant columns
df = df.loc[:, (df != df.iloc[0]).any()]

# #drop unnecessary variables
df = df.drop(['Adjusted r-squared','crspno','year','Data Date', 'Report Date of Quarterly Earnings','woGW', 'Delisting Return','Exchange Code'\
             ,'Shares Outstanding','Volume', 'Special Items','eamonth', 'grGW'], axis=1)

# # merge with the other one
welch = pd.read_excel('welch.xlsx')
welch['tms'] = welch['lty'] - welch['tbl']
welch['dfy'] = welch['BAA'] - welch['AAA']
welch['d/p'] = welch['D12'].apply(np.log) - welch['Index'].apply(np.log)
welch['e/p'] = welch['E12'].apply(np.log) - welch['Index'].apply(np.log)

welch = welch[['yyyymm','d/p','e/p','b/m','ntis','tbl','tms','dfy','svar']]

## they never merge it
df_m = df.merge(welch,how='inner',left_on='Date of Observation',right_on='yyyymm')
df_m = df_m.drop('yyyymm', axis =1)

for x in df:
  for y in welch:
      if x == 'Date of Observation' or x == 'Return' or x == 'compkey' or y == 'yyyymm':
        continue
      else:
        name = str(x) + str(y)
        df_m[name] = df_m[x] * df_m[y]

df = df_m

df = (df.fillna(method='ffill') + df.fillna(method='bfill'))/2

def getvals(id, lags):
  stock = df[df['compkey']== id]
  stock= stock.sort_values(by='Date of Observation')
  stock = stock.drop(['compkey','Date of Observation'], axis= 1)
  stock = stock.assign(Returns_lagged=stock['Returns'].shift(lags).values)
  stock = stock.drop(stock.index[lags:])
  stock = stock[[c for c in stock if c not in ['Returns', 'Returns_lagged']] + ['Returns', 'Returns_lagged']]
  vals = stock.values
  return vals

def splitsets(vals, train_perc,valid_perc, lstm= False):
  n_train = floor(vals.shape[0]*train_perc)
  n_val = floor(n_train + vals.shape[0]*valid_perc)
  train = vals[:n_train,:]
  valid = vals[n_train:n_val,:]
  test = vals[n_val+1:,:]
  train_X, train_y = train[:, :-1], train[:, -1]
  valid_X , valid_y = valid[:,:-1], valid[:,-1]
  test_X, test_y = test[:, :-1], test[:, -1]
  if lstm:
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    valid_X = valid_X.reshape((valid_X.shape[0],1, valid_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

  return train_X, train_y , valid_X, valid_y, test_X, test_y


# batch normalization, learning rate shrinkage, early stopping
def nn_build_compile(optimizer='Adam',train_X= np.zeros((375,944)), init_mode='glorot_uniform', reg_1 = 0.001, reg_2 = 0, learn= 0.01,\
                     n_epoch= 200,decay = 0, dropout = 0.1,activation_1 ='relu', activation_2='linear', num_layers = 3):
  model = Sequential()
  for i in range(num_layers):
    model.add(Dense(int(32/2**i),kernel_initializer = init_mode, kernel_regularizer = l1_l2(l1=reg_1, l2=reg_2),activation= activation_1))
    model.add(Dropout(dropout))
  if optimizer == 'Adam':
    optimizer = Adam(lr= learn)
  if optimizer == 'Adadelta':
    optimizer = Adadelta(lr= learn)
  if optimizer == 'Nadam':
    optimizer = Nadam(lr=learn)
  if optimizer == 'RMSprop':
    optimizer = RMSprop(lr=learn)
  model.compile(loss='mse', optimizer= optimizer, metrics=['accuracy'])
  return model

def lstm_build_compile(optimizer= 'Adam',train_X= np.zeros((375, 1, 944)), init_mode='glorot_uniform', reg_1 = 0.001, reg_2 = 0,\
                       learn= 0.01, n_epoch= 200,decay = 0, dropout = 0.1, activation_1 ='relu', activation_2='linear'):
  model = Sequential()
  model.add(LSTM(32,kernel_initializer = init_mode,input_shape = (train_X.shape[1],train_X.shape[2]),\
                 kernel_regularizer = l1_l2(l1=reg_1, l2=reg_2),return_sequences= True))
  model.add(Dropout(dropout))
  model.add(LSTM(16,kernel_initializer = init_mode,input_shape = (train_X.shape[1],train_X.shape[2]),\
                 kernel_regularizer = l1_l2(l1=reg_1, l2=reg_2), return_sequences= True ))
  model.add(Dropout(dropout))
  model.add(LSTM(8, input_shape = (train_X.shape[1], train_X.shape[2]), return_sequences = False))
  model.add(Dropout(dropout))
  model.add(Dense(1, kernel_initializer = init_mode, kernel_regularizer = l1_l2(l1=reg_1, l2=reg_2)))
  if optimizer == 'Adam':
    optimizer = Adam(lr= learn)
  if optimizer == 'Adadelta':
    optimizer = Adadelta(lr= learn)
  if optimizer == 'Nadam':
    optimizer = Nadam(lr=learn)
  if optimizer == 'RMSprop':
    optimizer = RMSprop(lr=learn)
  model.compile(loss='mse', optimizer= optimizer, metrics=['accuracy'])

  return model

def find_RMSE_plot(test_X, test_y,yhat, lstm= False):
  # invert scaling for forecast
  if lstm:
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
  inv_yhat = concatenate((test_X,yhat), axis=1) #test_X[:,1:]
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,-1]
  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  inv_y = concatenate((test_X,test_y), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,-1]
  # calculate RMSE
  rsquare = 1- np.sum(np.square(inv_y - inv_yhat))/ np.sum(np.square(inv_y))
  print('Test R-square: %.3f' % rsquare)

  plt.plot(inv_y, label='true value')
  plt.plot(inv_yhat, label='prediction')
  plt.legend()
  plt.show()

def plot_loss(history):
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()

## section for grid search
es = EarlyStopping(monitor='loss', verbose=1, patience = 100, restore_best_weights=True)
num_epoch = 400
train_perc = 0.825
valid_perc = 0.15
max_obs= [2049,1633,6821,4771,9882,1686,8850,1678,4737,9850,6781,3708, 10860]

for stock in max_obs:  # shift should be negative!
  vals = getvals(stock,-1)

    #scaling needs to be done outside
  n_train = floor(vals.shape[0]*train_perc)
  n_val = floor(n_train + vals.shape[0]*valid_perc)
  scaler = MinMaxScaler()
  vals[:n_val,:] = scaler.fit_transform(vals[:n_val,:])
  vals[n_val:,:] = scaler.transform(vals[n_val:,:])

  #LSTM
  train_X, train_y , valid_X, valid_y, test_X, test_y = splitsets(vals,train_perc,valid_perc, lstm = False)

  model = KerasRegressor(build_fn=nn_build_compile, epochs= num_epoch, verbose=0, shuffle=False)

  #Grid Search
#   init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#   #things to optimize: hidden layers, hidden neurons, activation, maybe optimizer
#   #think about running multiple networks like the paper
#   reg_1 = [0,0.001,0.01]
#   reg_2 = [0,0.001,0.01]
  layers = [1,2,3,4,5]
#   dropout = [0, 0.1, 0.2, 0.3, 0.4]
#   activation_1= ['relu','sigmoid','tanh','linear']
#   activation_2= ['relu','sigmoid','tanh','linear']
#   learn= [0.1,0.01,0.001,0.0001,0.00001]
#   optimizer = ['Adadelta','Adam', 'Nadam', 'RMSprop']
  param_grid = dict(num_layers=layers)
  grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring="neg_mean_squared_error", cv=3, n_jobs =-1)
  grid_result = grid.fit(train_X, train_y, callbacks= [es])
#   summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


es = EarlyStopping(monitor='val_loss', verbose=1, patience = 100, restore_best_weights=True)
train_perc = 0.825
n_epoch = 10000
reg_1 = 0.001
reg_2 = 0
learn = 0.1
valid_perc = 0.15
max_obs= [2049,1633,6821,4771,9882,1686,8850,1678,4737,9850,6781,3708, 10860]

for stock in max_obs:

  # shift should be negative!
  vals = getvals(stock,-1)

  #scaling needs to be done outside
  n_train = floor(vals.shape[0]*train_perc)
  n_val = floor(n_train + vals.shape[0]*valid_perc)
  scaler = MinMaxScaler()
  vals[:n_val,:] = scaler.fit_transform(vals[:n_val,:])
  vals[n_val:,:] = scaler.transform(vals[n_val:,:])

  #Vanilla NN
  train_X, train_y , valid_X, valid_y, test_X, test_y = splitsets(vals,train_perc,valid_perc, lstm = False)

  # design network
  model = nn_build_compile(train_X, n_epoch = n_epoch, reg_1 = reg_1, reg_2 = reg_2)

  # fit network
  history = model.fit(train_X, train_y, epochs= n_epoch, batch_size=32, validation_data=(valid_X, valid_y), verbose=0, shuffle=False, callbacks= [es])
  # plot history
  plot_loss(history)

  yhat = model.predict(test_X)

  find_RMSE_plot(test_X,test_y,yhat, lstm = False)

  #LSTM
  train_X, train_y , valid_X, valid_y, test_X, test_y = splitsets(vals,train_perc, valid_perc, lstm= True)

  # design network
  model = model_build_compile(train_X, lstm = True, reg_1 = reg_1)

  # fit network
  history = model.fit(train_X, train_y, epochs= n_epoch, batch_size=32, validation_data=(valid_X, valid_y), verbose=0, shuffle=False, callbacks= [es])
  # plot history
  plot_loss(history)

  # make a prediction
  yhat = model.predict(test_X)

  find_RMSE_plot(test_X,test_y,yhat, lstm = True)
