import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

model = load_model("stock_price_prediction_model.keras")

st.header("Stock Price Prediction App")
st.write(
    "This app predicts the stock prices of a company using historical data and a trained LSTM model.")
stock = st.text_input("Enter the stock ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = '2010-01-01'
end_date = '2025-01-01'

df = yf.download(stock, start=start_date, end=end_date)
st.subheader(f"Stock Data for {stock}")
df.columns = df.columns.droplevel(1)

st.write(df)

df_train = pd.DataFrame(df.Close[0:int(len(df)*0.8)])
df_test = pd.DataFrame(df.Close[int(len(df)*0.8):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

pass_100_days = df_train.tail(100)
data_test = pd.concat((pass_100_days, df_test), ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

x = []
y = []
for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i, 0])
    y.append(data_test_scaler[i, 0])

st.subheader("Price Vs MA50")
ma50_data = df.Close.rolling(window=50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma50_data, label='MA50')
plt.plot(df.Close, label='Close Price')
plt.xlabel('Years')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader("Price Vs MA50 Vs MA100")
ma100_data = df.Close.rolling(window=100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma100_data, label='MA100')
plt.plot(ma50_data, label='MA50')
plt.plot(df.Close, label='Close Price')
plt.xlabel('Years')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader("Price Vs MA100 Vs MA200")
ma200_data = df.Close.rolling(window=200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma200_data, label='MA200')
plt.plot(ma100_data, label='MA100')
plt.plot(df.Close, label='Close Price')
plt.xlabel('Years')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)

x , y = np.array(x), np.array(y)
predict = model.predict(x)

predict = predict * 1/scaler.scale_
y = y * 1/scaler.scale_

st.subheader("Predicted vs Actual Prices")
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, label='Actual Price')
plt.plot(predict, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

st.subheader("Model Performance Metrics")
rmse = np.sqrt(mean_squared_error(y, predict))
r2 = r2_score(y, predict)
st.write(f"Mean Squared Error: {rmse/np.mean(y):.2f}")
st.write(f"R-squared: {r2:.2f}")

