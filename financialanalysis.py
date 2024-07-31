# !pip install plotly

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
import plotly.graph_objs as go
from plotly.offline import iplot

def fetch_data(ticker):
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    try:
        data = yf.download(ticker, start="2010-01-01", end=end_date)
        stock = yf.Ticker(ticker)
        return data, stock
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None, None

def calculate_ratios(stock):
    ratios = {}
    try:
        stats = stock.info
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        financials = stock.financials

        ratios['pe_ratio'] = stats.get('trailingPE', np.nan)
        total_debt = balance_sheet.get('Total Liab', pd.Series([np.nan])).iloc[-1]
        total_equity = balance_sheet.get('Total Stockholder Equity', pd.Series([np.nan])).iloc[-1]
        ratios['de_ratio'] = total_debt / total_equity if total_equity != 0 else np.nan
        net_income = financials.get('Net Income', pd.Series([np.nan])).iloc[-1]
        ratios['roe'] = net_income / total_equity if total_equity != 0 else np.nan
        current_assets = balance_sheet.get('Total Current Assets', pd.Series([np.nan])).iloc[-1]
        current_liabilities = balance_sheet.get('Total Current Liabilities', pd.Series([np.nan])).iloc[-1]
        ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else np.nan
        ratios['dividend_yield'] = stats.get('dividendYield', np.nan)
        ebit = cash_flow.get('Total Cash From Operating Activities', pd.Series([np.nan])).iloc[-1]
        interest_expense = cash_flow.get('Interest Expense', pd.Series([0])).iloc[-1]
        ratios['interest_coverage'] = ebit / -interest_expense if interest_expense != 0 else np.nan
        total_sales = financials.get('Total Revenue', pd.Series([np.nan])).iloc[-1]
        average_assets = (balance_sheet.get('Total Assets', pd.Series([np.nan])).iloc[-1] + balance_sheet.get('Total Assets', pd.Series([np.nan])).iloc[0]) / 2
        ratios['asset_turnover'] = total_sales / average_assets if average_assets != 0 else np.nan
    except Exception as e:
        print(f"Error calculating ratios for {stock.ticker}: {e}")
    return ratios

def model_data(data):
    if data is None or data.empty:
        return None, None, None, float('nan')
    data['Day'] = range(len(data))
    X = data[['Day']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    rmse = sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return model, X, y, rmse

def plot_data(data, model, X, y):
    trace1 = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Prices')
    trace2 = go.Scatter(x=X.index, y=model.predict(X), mode='markers', name='Predicted Prices', marker=dict(color='red'))
    layout = go.Layout(title='Stock Price Prediction', xaxis=dict(title='Date'), yaxis=dict(title='Price'), showlegend=True)
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)

def forecast_future_prices(model, data, days=30):
    if model is None or data.empty:
        print("Model or data is not available for forecasting.")
        return None
    last_day = data['Day'].iloc[-1]
    future_days = np.arange(last_day + 1, last_day + days + 1)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days)
    X_future = pd.DataFrame(future_days, columns=['Day'], index=future_dates)
    predicted_prices = model.predict(X_future)
    return X_future.index, predicted_prices

def plot_future_data(data, future_dates, future_prices):
    trace1 = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices')
    trace2 = go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Forecasted Prices', line=dict(color='red', dash='dash'))
    layout = go.Layout(title='Stock Price Forecast', xaxis=dict(title='Date'), yaxis=dict(title='Price'), showlegend=True)
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)

def investment_recommendation(ratios, model, rmse):
    if model is None:
        return "Unable to generate recommendation due to insufficient data."
    financial_health_score = np.nanmean([np.tanh(value) for value in ratios.values()])
    confidence = 100 * np.tanh(model.feature_importances_[0]) * financial_health_score if not np.isnan(financial_health_score) else 50
    recommendation = "Buy" if confidence > 50 else "Sell"
    return f"Investment Recommendation: {recommendation} with {confidence:.2f}% confidence, RMSE of model: {rmse:.2f}."

def run_analysis(ticker):
    data, stock = fetch_data(ticker)
    if data is not None and not data.empty:
        ratios = calculate_ratios(stock)
        model, X, y, rmse = model_data(data)
        plot_data(data, model, X, y)
        recommendation = investment_recommendation(ratios, model, rmse)
        future_dates, future_prices = forecast_future_prices(model, data, days=60)
        plot_future_data(data, future_dates, future_prices)
        return recommendation
    else:
        return f"No data to analyze for {ticker}"

tickers = ['ITC.NS']
for ticker in tickers:
    print(run_analysis(ticker))

