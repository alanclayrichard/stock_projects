from pyexpat import model
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np
from fredapi import Fred
from datetime import date

# Get todays date
day = date.today()
today = day.strftime("%Y-%m-%d")

class Analysis:
    # Polynomial Regression (any order polynomial up until overflow error ~90 order)
    def get_mult_linreg(X_train,Y_train,order,X_test):
        y = np.reshape(Y_train,[len(Y_train),1])
        x = np.reshape(X_train,[len(X_train),1])
        if order == 1:
            x_ones = np.ones([len(X_train),1])
            x = np.hstack((x_ones,x))
        else: 
            x_ones = np.ones([len(X_train),1])
            for i in range(1,order+1):
                x_ones = np.hstack((x_ones,x**i))
            x = x_ones
        Betas = np.dot(np.linalg.inv(np.dot(x.transpose(1,0),x)),np.dot(x.transpose(1,0),y))
        if order == 1:
            mult_linreg_mdl = Betas[0] + Betas[1]*X_test
        else:
            mult_linreg_mdl = Betas[0]
            for j in range(1,order+1):
                mult_linreg_mdl = mult_linreg_mdl + Betas[j]*(X_test**j)
        return mult_linreg_mdl

    # Exponential Regression
    def get_exp_linreg(X_train,Y_train,X_test):
        log_prices = np.log(Y_train[1:len(Y_train)])
        log_index = X_train[1:len(X_train)]
        x_loghat = np.average(log_index)
        y_loghat = np.average(log_prices)
        log_beta1 = np.sum((log_prices-y_loghat)*(log_index-x_loghat))/np.sum((log_index-x_loghat)**2)
        log_beta0 = y_loghat - log_beta1*x_loghat
        exp_mdl = np.exp(log_beta1*(X_test[1:len(X_test)]) + log_beta0)
        return exp_mdl
    
    # Plot general regression model
    def plot_regmdl(x_train,y_train,x_test,y_test,model_type):
        if model_type == "exp":
            plt.figure(figsize=[14,6])
            plt.plot(x_train[1:len(x_train)],y_train[1:len(x_train)])
            plt.plot(x_test[1:len(x_test)],y_test,"--",linewidth=1)
            plt.title(model_type+" Analysis")
            plt.legend(["training data","test data"])
            plt.show()
        else:
            plt.figure(figsize=[14,6])
            plt.plot(x_train,y_train)
            plt.plot(x_test,y_test,"--",linewidth=1)
            plt.title(str(model_type)+" Order Analysis")
            plt.legend(["training data","test data"])
            plt.show()

class Stock:
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

    # Perform and show regression for given ticker, time, etc
    def show_stockreg(ticker,time_period,future_days,model_type):
        data,x_train,x_test = Stock.get_data(ticker,time_period,future_days)
        if model_type == "exp":
            mdl = Analysis.get_exp_linreg(x_train,data,x_test)
            Analysis.plot_regmdl(x_train,data,x_test,mdl,model_type)
        else: 
            mdl = Analysis.get_mult_linreg(x_train,data,model_type,x_test)
            Analysis.plot_regmdl(x_train,data,x_test,mdl,model_type)

    # Compare Two Plots with Polynomial Regression
    def compare_two_plot(ticker1,ticker2,time_period,future_days,order=1):
        close_prices1,index1,pred_index1 = Stock.get_data(ticker1,time_period,future_days)
        close_prices2,index2,pred_index2 = Stock.get_data(ticker2,time_period,future_days)
        mdl1 = Analysis.get_mult_linreg(index1,close_prices1,order,pred_index1)
        mdl2 = Analysis.get_mult_linreg(index2,close_prices2,order,pred_index2)
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
        close_prices,index,pred_index = Stock.get_data(ticker,time_period,future_days)
        plt.figure(figsize=[14,6])
        plt.plot(index,close_prices)
        legend_names = [ticker + " price"]
        for i in range(1,max_order+1):
            mdl = Analysis.get_mult_linreg(index,close_prices,i,pred_index)
            plt.plot(pred_index,mdl,"--",linewidth=.75)
            legend_names = np.hstack((legend_names,"order "+str(i)))
        exp_mdl = Analysis.get_exp_linreg(index,close_prices,pred_index)
        plt.plot(pred_index[1:len(pred_index)],exp_mdl,"--",linewidth=.75)
        legend_names = np.hstack((legend_names,"exp"))
        plt.legend(legend_names)
        plt.title(ticker +" Regression Models up to order "+str(max_order)+ " and exponential")
        plt.ylabel("share price")
        plt.xlabel("previous " + time_period)
        plt.show()

class Crypto:
    # Get historical Bitcoin prices
    def get_bitcoin_price(start_date,end_date=today):
        FRED_API_KEY = '8da06254f69eb0b6a6a0517042bb43f4'
        fred = Fred(api_key=FRED_API_KEY)
        data = fred.get_series('CBBTCUSD',start_date+' 00:00:00',end_date+' 00:00:00').to_numpy()
        index = np.linspace(0,len(data),len(data))
        return data, index

class Govt:
    # Get federal funds interest rate from FRED data
    def get_DFF(start_date,end_date=today):
        FRED_API_KEY = '8da06254f69eb0b6a6a0517042bb43f4'
        fred = Fred(api_key=FRED_API_KEY)
        data = fred.get_series('DFF',start_date+' 00:00:00',end_date+' 00:00:00').to_numpy()
        index22 = np.linspace(0,len(data),len(data))
        return data, index22

# Perform the analysis:
# Stock.show_stockreg("AAPL","10y","10",9)

Stock.plot_all_mdls("SPY","10y","100",9)

Bit_price = Crypto.get_bitcoin_price("2021-02-21")
print(Bit_price)