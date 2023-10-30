# dashboard development for soligence using streamlit applicattion
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy as np

# Page layout
st.set_page_config(layout="wide")

# Title
st.title('Trading Dashboard')
st.markdown("""
This app retrieves cryptocurrency prices for the top 20 cryptocurrencies from the **Yfinance**!
""")


# About
expander_bar = st.expander("About")
expander_bar.markdown("""
Solent intelligence (SOLiGence) is a leading financial multinational organisation that deals with stocks, shares, savings and investments.
""")

## Divide page to 2 columns (col1 = sidebar, col2 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2,1))


crypto_symbols = ['BTC','ETH', 'LTC', 'DOGE', 'PPC', 'ADA', 'USDT', 'NXT', 'XRP', 'XPM',
                 'GRC', 'AUR', 'VTC', 'XLM', 'XVG', 'BCH', 'XMR', 'FIRO', 'DASH', 'NEO']

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

#--------------------------------------------------------------------------------------------------------#
col1.header('Input Options')

## Sidebar - Currency price unit
currency_price_unit = col1.selectbox('Select currency for price', tuple(crypto_symbols))

predict_range = col1.slider('Number of Days', 1, 30)

user_amount = col1.text_input("Enter Amount to Invest(USD)", 0)

selected_coin = col1.multiselect('Coin Performance Relationship', crypto_symbols, crypto_symbols[:10])

today = date.today()
d1 = today.strftime("%Y-%m-%d")

crypto_data = yf.download([f'{currency_price_unit}-USD'], start='2014-01-01', end=d1, interval='1d')

@st.cache
def predict_model(coin, timeline):
    coin_data = yf.download([f'{coin}-USD'], start='2014-01-01', end=d1, interval='1d')
    coin_data = coin_data.dropna()
    coin_data = coin_data.reset_index()
    coin_data_close = coin_data['Close']
    close_price = np.array(coin_data_close).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_price)

    ##splitting dataset into train and test split
    training_size = int(len(scaled_close) * 0.75)
    test_size = len(scaled_close) - training_size
    train_data, test_data = scaled_close[0:training_size, :], scaled_close[training_size:len(scaled_close), :1]
    try:
        time_step = 100
        X_test, ytest = create_dataset(test_data, time_step)

        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        output = []
        n_steps = 100
        i = 0
        saved_model = load_model(f'trained_model/{coin}')
        scores = saved_model.evaluate(X_test, ytest)
        LSTM_accuracy = (1 - scores) * 100

        while (i < timeline):
            if (len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = saved_model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = saved_model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                i = i + 1
        return output, LSTM_accuracy
    except IOError as e:
        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)
        model.save(f'{coin}')

        scores = model.evaluate(X_test, ytest)

        LSTM_accuracy = (1 - scores) * 100
        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        output = []
        n_steps = 100
        i = 0

        while (i < timeline):
            if (len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0])
                temp_input.extend(yhat[0].tolist())
                i = i + 1
        return output, LSTM_accuracy

predictions, accuracy = predict_model(currency_price_unit, int(predict_range))

col2.write(pd.DataFrame({
    f'{currency_price_unit}-Predictions': predictions
}))
col3.subheader('Prediction Accuracy')
col3.write(f'Model Accuracy: {accuracy}%', font = 12, color = 'Green')

def get_profit(values, investment):
    track = 0
    k = 0

    while k < len(values) - 1:
        track += investment * ((values[k+1] - values[k]) / values[k])
        k = k+1
    return round(track,5)
col3.subheader('Profit for Investment')
col3.write(f'Your investment profit for chosen days is : {get_profit(predictions, float(user_amount))}', font = 20)

@st.cache
def coin_performance(coins):
    ticker = []
    for coin in coins:
        ticker.append(f'{coin}-USD')
    dataset = yf.download(ticker, start="2014-01-01", end=d1)

    dataset = dataset.loc[:, "Close"].copy().dropna()
    return dataset


#-------------------------------------------------------------------------------------------------------------
#Visualizations
col2.subheader(f' {currency_price_unit} Closing Price vs Time Chart')
fig = plt.figure(figsize = (6,2))
plt.plot(crypto_data.Close)
col2.pyplot(fig)

#plot ma100
col2.subheader(f'{currency_price_unit} Closing Price vs Time Chart with 100MA')
ma100= crypto_data.Close.rolling(100).mean()
fig = plt.figure(figsize = (6,2))
plt.plot(ma100)
plt.plot(crypto_data.Close)
col2.pyplot(fig)

#heat map showing correlated values
col2.subheader('Positive and Negative Correlated Cryptocurrencies')
plt.figure(figsize=(12,9))
sns.set(font_scale=1.5)
sns.heatmap(coin_performance(selected_coin).corr(), cmap = "YlGnBu", annot = True, annot_kws={"size":15}, vmax=1)
col2.pyplot(plt)

# #plot original vs predicted
# plt.figure(figsize=(12,6))
# plt.plot(ytest, 'b', label = 'Original Price')
# plt.plot(predictions, 'r', label = 'Predicted Price')
# plt.x_label('Time')
# plt.y_label('Price')
# plt.show()
