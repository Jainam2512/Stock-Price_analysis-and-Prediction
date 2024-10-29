import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go

# Streamlit app title and sidebar
st.title('Enhanced Stock Analysis and Prediction')
st.sidebar.info('Welcome to the Enhanced Stock Analysis App! Created by Jainam Shah.')

# Main function to control the app
def main():
    option = st.sidebar.selectbox('Choose an Option', ['Visualize Indicators', 'Recent Data', 'Predict'])
    if option == 'Visualize Indicators':
        tech_indicators()
    elif option == 'Recent Data':
        display_data()
    else:
        predict_stock_price()

@st.cache_resource
def download_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return df

# Sidebar inputs for stock selection and date range
symbol = st.sidebar.text_input('Stock Symbol', value='AAPL').upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Duration (days)', value=3000)
start_date = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=start_date)
end_date = st.sidebar.date_input('End Date', value=today)
data = download_data(symbol, start_date, end_date)
scaler = StandardScaler()

# Function to display technical indicators
def tech_indicators():
    st.header(f'Technical Indicators for {symbol}')
    indicator_option = st.radio('Select Indicator', ['Close Price', 'Bollinger Bands', 'MACD', 'RSI', 'ATR', 'OBV'])

    # Ensure there are enough data points
    if data['Close'].isnull().any() or len(data) < 14:
        st.error("Insufficient data to calculate indicators. Please choose a different stock or adjust the date range.")
        return

    # Calculate indicators
    bb = BollingerBands(data['Close'])
    macd = MACD(data['Close'])
    rsi = RSIIndicator(data['Close'])
    atr = AverageTrueRange(data['High'], data['Low'], data['Close'])
    obv = OnBalanceVolumeIndicator(data['Close'], data['Volume'])
    sma = SMAIndicator(data['Close'], window=20)
    ema = EMAIndicator(data['Close'], window=20)

    # Initialize Plotly figure
    fig = go.Figure()

    # Plot selected indicator
    if indicator_option == 'Close Price':
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f'{symbol} Closing Prices', xaxis_title='Date', yaxis_title='Price (USD)')
    elif indicator_option == 'Bollinger Bands':
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=data.index, y=bb.bollinger_hband(), mode='lines', name='Upper Band', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=bb.bollinger_lband(), mode='lines', name='Lower Band', line=dict(color='green', dash='dash')))
        fig.update_layout(title='Bollinger Bands Analysis', xaxis_title='Date', yaxis_title='Price (USD)')
    elif indicator_option == 'MACD':
        fig.add_trace(go.Scatter(x=data.index, y=macd.macd(), mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=macd.macd_signal(), mode='lines', name='Signal Line'))
        fig.add_trace(go.Bar(x=data.index, y=macd.macd_diff(), name='MACD Histogram', marker_color='gray'))
        fig.update_layout(title='MACD Analysis', xaxis_title='Date', yaxis_title='MACD Value')
    elif indicator_option == 'RSI':
        rsi_values = rsi.rsi()
        fig.add_trace(go.Scatter(x=data.index, y=rsi_values, mode='lines', name='RSI'))
        fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought', annotation_position='bottom right')
        fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold', annotation_position='top right')
        fig.update_layout(title='RSI Analysis', xaxis_title='Date', yaxis_title='RSI Value')
    elif indicator_option == 'ATR':
        fig.add_trace(go.Scatter(x=data.index, y=atr.average_true_range(), mode='lines', name='ATR'))
        fig.update_layout(title='Average True Range (ATR)', xaxis_title='Date', yaxis_title='ATR Value')
    elif indicator_option == 'OBV':
        fig.add_trace(go.Scatter(x=data.index, y=obv.on_balance_volume(), mode='lines', name='OBV'))
        fig.update_layout(title='On-Balance Volume (OBV)', xaxis_title='Date', yaxis_title='Volume')

    # Interactive enhancements
    fig.update_layout(xaxis_rangeslider_visible=True, hovermode='x unified')
    st.plotly_chart(fig)

# Function to display recent data
def display_data():
    st.header(f'Recent Data for {symbol}')
    st.dataframe(data.tail(10))

# Prediction function
def predict_stock_price():
    model_choice = st.radio('Choose a Model', ['Linear Regression', 'Random Forest', 'Extra Trees', 'K-Nearest Neighbors', 'XGBoost', 'Ensemble'])
    forecast_days = int(st.number_input('Days to Predict', value=5))

    if st.button('Predict'):
        model = choose_model(model_choice)
        perform_prediction(model, forecast_days)

# Function to choose model based on user's choice
def choose_model(choice):
    if choice == 'Linear Regression':
        return LinearRegression()
    elif choice == 'Random Forest':
        return RandomForestRegressor()
    elif choice == 'Extra Trees':
        return ExtraTreesRegressor()
    elif choice == 'K-Nearest Neighbors':
        return KNeighborsRegressor()
    elif choice == 'XGBoost':
        return XGBRegressor()
    else:
        # Ensemble model for better predictions
        return VotingRegressor([
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor()),
            ('xgb', XGBRegressor())
        ])

# Function to perform prediction
def perform_prediction(model, days):
    df = data[['Close']].copy()
    df['Return'] = df['Close'].pct_change()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df.dropna(inplace=True)

    # Prepare the data
    X = scaler.fit_transform(df[['Close', 'Return', 'Lag1', 'Lag2']].values[:-days])
    y = df['Close'].values[2: -days+2]
    X_forecast = scaler.transform(df[['Close', 'Return', 'Lag1', 'Lag2']].values[-days:])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display evaluation metrics
    st.text(f'R² Score: {r2_score(y_test, predictions):.4f}')
    st.text(f'Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}')

    # Future predictions
    future_predictions = model.predict(X_forecast)

    # Generate future trading dates, excluding weekends and holidays
    future_dates = pd.bdate_range(start=data.index[-1], periods=days + 1, closed='right')

    # Create a DataFrame for future predictions with dates
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
    st.subheader(f'{symbol} Predicted Prices for Next {days} Trading Days')
    st.dataframe(future_df)

    # Plotting the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Predicted Prices', line=dict(color='blue', dash='dash')))
    fig.update_layout(title=f'{symbol} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price (USD)', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Run the app
if __name__ == '__main__':
    main()
