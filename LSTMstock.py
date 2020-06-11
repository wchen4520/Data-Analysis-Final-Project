# In[0]

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings;warnings.filterwarnings('ignore')


# In[1]

df = pd.read_csv('C:\\data\\SPYrow_new.csv').drop(["Date"], axis=1)

df_X = df[['High', 'Low', 'Open', 'Close', 'Volume']]
df_X = df_X.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
df_y = df[['Close']]
df_y = df_y.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))


# In[2]

def buildTrain(df_X, df_y, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(df_X.shape[0]-futureDay-pastDay):
        X_train.append(np.array(df_X.iloc[i:i+pastDay]))
        Y_train.append(np.array(df_y.iloc[i+pastDay:i+pastDay+futureDay]["Close"]))
    return np.array(X_train), np.array(Y_train)
    

# In[3]  input: 30days  output: 1day
    
X, y = buildTrain(df_X, df_y, 10, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences = False))
model.add(Dropout(0.1))
model.add(Dense(units = 16, init = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, init = 'uniform', activation = 'linear'))
model.compile(loss='mse', optimizer="adam")  #val_accuracy, metrics = ['accuracy']
callback = EarlyStopping(monitor="val_loss", patience=10, mode="auto")
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_val, y_val), callbacks=[callback])  
model.summary()

    
# In[4]

pred = DataFrame(model.predict(X_test))

plt.plot(y_test, color = 'red', label = 'Real SPY Price')  # 紅線表示真實股價
plt.plot(pred, color = 'blue', label = 'Predicted SPY Price')  # 藍線表示預測股價
plt.title('SPY Price Prediction')
plt.xlabel('Time')
plt.ylabel('SPY Price')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, pred)
Rsquare = 1-mse/np.var(y_test)
print('R-square = ', Rsquare)

    
# In[5]

