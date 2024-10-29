import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set title
st.title("Stock Price Prediction App")

# User input for stock symbol and prediction days
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
days = st.number_input("Enter number of days to predict:", min_value=1, max_value=30, value=5)

# Fetch data
@st.cache
def fetch_data(symbol):
    data = yf.download(symbol, period="1y")
    return data

data = fetch_data(symbol)

# Display fetched data
st.subheader(f"Data for {symbol}")
st.dataframe(data)

# Prepare data for prediction
def prepare_data(data):
    data['Return'] = data['Close'].pct_change()
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data.dropna(inplace=True)
    
    X = data[['Lag1', 'Lag2']]
    y = data['Close']
    return X, y

X, y = prepare_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Future predictions
future_lag1 = data['Close'].iloc[-1]
future_lag2 = data['Close'].iloc[-2]
future_data = pd.DataFrame([[future_lag1, future_lag2]], columns=['Lag1', 'Lag2'])
predictions = []

for _ in range(days):
    prediction = model.predict(future_data)
    predictions.append(prediction[0])
    future_lag2 = future_lag1
    future_lag1 = prediction[0]
    future_data = pd.DataFrame([[future_lag1, future_lag2]], columns=['Lag1', 'Lag2'])

# Create future dates
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Prices'))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'], mode='lines+markers', name='Predicted Prices', line=dict(color='blue', dash='dash')))

# Update layout
fig.update_layout(title=f'{symbol} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price (USD)', xaxis_rangeslider_visible=True)

# Display the figure
st.plotly_chart(fig)

# Show predicted prices
st.subheader(f'{symbol} Predicted Prices for Next {days} Days')
st.dataframe(future_df)
