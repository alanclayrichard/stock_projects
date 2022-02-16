from codecs import IncrementalDecoder
from operator import index
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import requests 
import nasdaqdatalink
from fredapi import Fred
from datetime import date

day = date.today()
today = day.strftime("%Y-%m-%d")

# Get Data for Tickers
def get_data(ticker,time_period,days_future='10'):
    data = yf.Ticker(ticker)
    time_data = data.history(period=time_period)
    # data will be closing prices for this model
    close_prices = time_data.iloc[:,3].to_numpy()
    # create an index to regress onto and plot and then create a prediction index to model future behavior (x) days into the future
    index = np.linspace(0,len(close_prices),len(close_prices))
    pred_index = np.linspace(0,len(close_prices)+int(days_future),len(close_prices)+int(days_future))
    return close_prices,index,pred_index

# Get Current Prices
def print_current_price(ticker):
    data = yf.Ticker(ticker)
    time_data = data.history(period="10d")
    close_prices = time_data.iloc[:,3].to_numpy()
    tick_text = ticker + " is at $" + str(close_prices[-1]) + " per share most recently"
    print(tick_text)

# Market Cap
def print_mkt_cap(ticker):
    data = yf.Ticker(ticker)
    ticker_marketCap = data.info['marketCap']
    print(ticker + ' Market Cap: $'+ str(ticker_marketCap) + " most recently")

# Polynomial Regression (any order polynomial up until overflow error ~90 order)
def get_poly_linreg(X,Y,order,X_prediction):
    y = np.reshape(Y,[len(Y),1])
    x = np.reshape(X,[len(X),1])
    if order == 1:
        x_ones = np.ones([len(X),1])
        x = np.hstack((x_ones,x))
    else: 
        x_ones = np.ones([len(X),1])
        for i in range(1,order+1):
            x_ones = np.hstack((x_ones,x**i))
        x = x_ones
    Betas = np.dot(np.linalg.inv(np.dot(x.transpose(1,0),x)),np.dot(x.transpose(1,0),y))
    if order == 1:
        poly_linreg_mdl = Betas[0] + Betas[1]*X_prediction
    else:
        poly_linreg_mdl = Betas[0]
        for j in range(1,order+1):
            poly_linreg_mdl = poly_linreg_mdl + Betas[j]*(X_prediction**j)
    return poly_linreg_mdl

# Exponential Regression
def get_exp_linreg(X,Y,X_prediction):
    log_prices = np.log(Y[1:len(Y)])
    log_index = X[1:len(X)]
    x_loghat1 = np.average(log_index)
    y_loghat1 = np.average(log_prices)
    log_beta1 = np.sum((log_prices-y_loghat1)*(log_index-x_loghat1))/np.sum((log_index-x_loghat1)**2)
    log_beta0 = y_loghat1 - log_beta1*x_loghat1
    exp_mdl = np.exp(log_beta1*(X_prediction[1:len(X_prediction)]) + log_beta0)
    return exp_mdl

# Compare Two Plots with Polynomial Regression
def compare_two_plot(ticker1,ticker2,time_period,future_days,order=1):
    close_prices1,index1,pred_index1 = get_data(ticker1,time_period,future_days)
    close_prices2,index2,pred_index2 = get_data(ticker2,time_period,future_days)
    mdl1 = get_poly_linreg(index1,close_prices1,order,pred_index1)
    mdl2 = get_poly_linreg(index2,close_prices2,order,pred_index2)
    plt.figure(figsize=[14,6])
    plt.plot(index1,close_prices1,zorder=7)
    plt.plot(index2,close_prices2,zorder=8)
    plt.plot(pred_index1,mdl1,"--",linewidth=.75)
    plt.plot(pred_index2,mdl2,"--",linewidth=.75)
    plt.title(ticker1 + " and " + ticker2 + " Analysis")
    plt.ylabel("share price")
    plt.xlabel("previous " + time_period)
    plt.legend([ticker1,ticker2,ticker1+ " " + str(order) + " order model",ticker1+ " " + str(order) + " order model"])

# Plot all of the Linear Regression Models for Stock prices (including exponenital)
def plot_all_mdls(ticker,time_period,future_days,max_order=3):
    close_prices,index,pred_index = get_data(ticker,time_period,future_days)
    plt.figure(figsize=[14,6])
    plt.plot(index,close_prices)
    legend_names = [ticker + " price"]
    for i in range(1,max_order+1):
        mdl = get_poly_linreg(index,close_prices,i,pred_index)
        plt.plot(pred_index,mdl,"--",linewidth=.75)
        legend_names = np.hstack((legend_names,"order "+str(i)))
    exp_mdl = get_exp_linreg(index,close_prices,pred_index)
    plt.plot(pred_index[1:len(pred_index)],exp_mdl,"--",linewidth=.75)
    legend_names = np.hstack((legend_names,"exp"))
    plt.legend(legend_names)
    plt.title(ticker +" Regression Models up to order "+str(max_order)+ " and exponential")
    plt.ylabel("share price")
    plt.xlabel("previous " + time_period)

def get_DFF(start_date,end_date=today):
    FRED_API_KEY = '8da06254f69eb0b6a6a0517042bb43f4'
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series('DFF',start_date+' 00:00:00',end_date+' 00:00:00').to_numpy()
    index22 = np.linspace(0,len(data),len(data))
    return data, index22

def get_bitcoin_price(start_date,end_date=today):
    FRED_API_KEY = '8da06254f69eb0b6a6a0517042bb43f4'
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series('CBBTCUSD',start_date+' 00:00:00',end_date+' 00:00:00').to_numpy()
    index = np.linspace(0,len(data),len(data))
    return data, index

# Get the data for the tickers defined
# print_current_price("AAPL")
# print_current_price("GOOG")
# print_mkt_cap("AAPL")
# print_mkt_cap("GOOG")

# compare_two_plot("AAPL","GOOG","10y","100",5)
# plt.show()

# plot_all_mdls("goog","10y","100",5)
# plt.show()

# Get Federal Funds Interest rate
fed_funds_rate,index_ff = get_DFF('2001-02-15')

# # Regress interest rate onto its index
mdl = get_poly_linreg(index_ff,fed_funds_rate,6,index_ff)

BTC,BTC_index = get_bitcoin_price('2010-01-01')

plt.plot(BTC_index,BTC)
plt.plot(index_ff,fed_funds_rate)
plt.plot(index_ff,mdl)
plt.show()