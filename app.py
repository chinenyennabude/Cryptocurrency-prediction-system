from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import yfinance as yf

app = Flask(__name__) #Initialize the flask App

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)
 
@app.route('/prediction')
def prediction():
 	return render_template("prediction.html")
    
@app.route('/predict',methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    coin = int_feature[0]
    days = int(int_feature[1])

    from datetime import date

    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%Y-%m-%d")
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
        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        output = []
        n_steps = 100
        i = 0
        saved_model = load_model(f'trained_model/{coin}')
        while (i < days):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = saved_model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = saved_model.predict(x_input, verbose=0)
                print(scaler.inverse_transform(yhat))
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                i = i + 1
        return render_template('prediction.html', prediction_text=output[days-1])
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

        x_input = test_data[len(test_data) - 100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        output = []
        n_steps = 100
        i = 0

        while (i < days):
            if (len(temp_input) > 100):
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=0)
                output.append(scaler.inverse_transform(yhat)[0][0])
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(scaler.inverse_transform(yhat))
                output.append(scaler.inverse_transform(yhat)[0])
                temp_input.extend(yhat[0].tolist())
                i = i + 1

        return render_template('prediction.html', prediction_text= output[days-1])

#@app.route('/chart')
#def chart():
	#abc = request.args.get('news')	
	#input_data = [abc.rstrip()]
	# transforming input
	#tfidf_test = tfidf_vectorizer.transform(input_data)
	# predicting the input
	#y_pred = pac.predict(tfidf_test)
    #output=y_pred[0]
	#return render_template('chart.html', prediction_text='Review is {}'.format(y_pred[0])) 

 

if __name__=='__main__':
    app.run(debug=True)
