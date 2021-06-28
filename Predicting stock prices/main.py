from itertools import Predicate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# This project is going to predict in the future for only the next day, so please do not use this project as referrence for the stock market (neural network in lstm network)
# Loading the data
company = 'FB'
start = dt.datetime(2012, 1, 1)  # From what date you want to collect the data
end = dt.datetime(2020, 1, 1)  # To where you want the data to stop

data = web.DataReader(company, 'yahoo', start, end)


# prepare the data
scaler = MinMaxScaler(feature_range=(0, 1))

# (Line 23) Predicting the price after the markte has closed
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


prediction_days = 60  # how many days do I want to base my prediciton on

x_train = []
y_train = []

# (Line 32) Start conting from index(60), and were going to go until the last index
for x in range(prediction_days, len(scaled_data)):
    # (Line 34) Add values to the x_traing with each iteration
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# (Line 38)Converting the trains into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build the model
model = Sequential()

# (Line 47) Adding add a LSDM layer, then a dropout layer and so on
# (Line 47)you can change the units, maybe less units=more better performance, The more units, the longer you have to train
model.add(LSTM(units=50, return_sequences=True,
               input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0, 2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0, 2))
model.add(LSTM(units=50))
model.add(Dropout(0, 2))
model.add(Dense(units=1))  # prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
# (line 59) Epochs= the model is going to see the data 24 times.  batch_size= the model is going to see 32 units at once
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test the model accuracy on existing data'''

# Load test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# (Line 72)What our moel will see as input
model_inputs = total_dataset[len(
    total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, x_test.shape[0], x_test.shape[1], 1)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# Predict next day

real_data = [
    model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
