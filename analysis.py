from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
import yfinance as yf
from fredapi import Fred
from datetime import date

# Get todays date
day = date.today()
today = day.strftime("%Y-%m-%d")

class Analysis:
    # Polynomial Regression (any order polynomial up until overflow error ~90 order)
    def get_mult_linreg(x_train: npt.NDArray, y_train: npt.NDArray, order: int, x_test: npt.NDArray = False) -> npt.NDArray:
        if isinstance(x_test,bool) == True:
            x_test = x_train
        Analysis.checknan(x_train)
        Analysis.checknan(x_train)
        Analysis.checknan(x_test)
        y = np.reshape(y_train,[len(y_train),1])
        x = np.reshape(x_train,[len(x_train),1])
        if order == 1:
            x_ones = np.ones([len(x_train),1])
            x = np.hstack((x_ones,x))
        else: 
            x_ones = np.ones([len(x_train),1])
            for i in range(1,order+1):
                x_ones = np.hstack((x_ones,x**i))
            x = x_ones
        Betas = np.dot(np.linalg.inv(np.dot(x.transpose(1,0),x)),np.dot(x.transpose(1,0),y))
        if order == 1:
            mult_linreg_mdl = Betas[0] + Betas[1]*x_test
        else:
            mult_linreg_mdl = Betas[0]
            for j in range(1,order+1):
                mult_linreg_mdl = mult_linreg_mdl + Betas[j]*(x_test**j)
        return mult_linreg_mdl

    # Exponential Regression
    def get_exp_linreg(x_train: npt.NDArray, y_train: npt.NDArray, x_test: npt.NDArray) -> npt.NDArray:
        log_prices = np.log(y_train[1:len(y_train)])
        log_index = x_train[1:len(x_train)]
        x_loghat = np.average(log_index)
        y_loghat = np.average(log_prices)
        log_beta1 = np.sum((log_prices-y_loghat)*(log_index-x_loghat))/np.sum((log_index-x_loghat)**2)
        log_beta0 = y_loghat - log_beta1*x_loghat
        exp_mdl = np.exp(log_beta1*(x_test[1:len(x_test)]) + log_beta0)
        return exp_mdl
    
    # Get r^2 value for a set of data points x and y 
    def get_r2(x_train: npt.NDArray, y_train: npt.NDArray) -> float:
        yhat = np.mean(y_train)
        ybar = Analysis.get_mult_linreg(x_train,y_train,1)
        rss = np.sum((y_train-ybar)**2)
        tss = np.sum((y_train-yhat)**2)
        r_sq = 1 - rss/tss
        return r_sq

    # Display Correlation on a plot (R^2 value displayed)
    def show_correlaition(x_train: npt.NDArray, y_train: npt.NDArray) -> None:
        r2 = Analysis.get_r2(x_train,y_train)
        xmax = np.max(np.abs(x_train))
        ymax = np.max(np.abs(y_train))
        if xmax>ymax:
            max = xmax
        else:
            max = ymax
        plt.scatter(x_train,y_train,c='red',s=4)
        plt.xlabel(str())
        plt.ylabel(str())
        x = np.linspace(-max,max,1000)
        plt.plot(x,x,"--",linewidth=.75)
        plt.annotate("R^2 = "+str(round(r2,3)),xy=(0,0),xytext=(-max,max))
        plt.show()
    
    # Plot general regression model
    def plot_regmdl(x_train: npt.NDArray, y_train: npt.NDArray, x_test: npt.NDArray, y_test: npt.NDArray, model_type: int) -> None:
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

    # Remove NaN values 
    def rmvnan(array: npt.NDArray) -> npt.NDArray:
        array = array[~np.isnan(array)]
        return array
    
    # Check for NaN data
    def checknan(array: npt.NDArray,name: str = "your stinky data") -> bool:
        k = len(np.shape(array))
        flag = False
        if k == 1:
            r = np.shape(array)[0]
            nan_bool = np.isnan(array)
            for i in range(0,r):
                if nan_bool[i]:
                    flag = True
                    print("warning: NaN detected at ["+str(i)+"] in " + name)
        elif k == 2:
            r = np.shape(array)[0]
            c = np.shape(array)[1]
            nan_bool = np.isnan(array)
            for i in range(0,r):
                for j in range(0,c):
                    if nan_bool[i,j]:
                        flag = True
                        print("warning: NaN detected at [" + str(i) + "," +str(j)+"] in " + name)
        return flag

class Stock:
    # Get Data for Tickers
    def get_data(ticker: str, time_period: str, days_future: str ='10') -> npt.NDArray:
        data = yf.Ticker(ticker)
        if Analysis.checknan(data) == True:
            data = Analysis.rmvnan(data)
            print("removed nan")
        time_data = data.history(period=time_period)
        # data will be closing prices for this model
        close_prices = time_data.iloc[:,3].to_numpy()
        # create an index to regress onto and plot and then create a prediction index to model future behavior (x) days into the future
        index = np.linspace(0,len(close_prices),len(close_prices))
        pred_index = np.linspace(0,len(close_prices)+int(days_future),len(close_prices)+int(days_future))
        return close_prices,index,pred_index

    # Get Current Prices
    def print_current_price(ticker: str) -> str:
        data = yf.Ticker(ticker)
        time_data = data.history(period="10d")
        close_prices = time_data.iloc[:,3].to_numpy()
        tick_text = ticker + " is at $" + str(close_prices[-1]) + " per share most recently"
        print(tick_text)

    # Get daily percent change
    def get_pct(ticker: str, time_period: str) -> npt.NDArray:
        close_price= Stock.get_data(ticker,time_period,'0')[0]
        day_change  = np.diff(close_price)
        pct_change = 100*(day_change/close_price[:-1])
        return pct_change

    # Market Cap
    def print_mkt_cap(ticker: str) -> str:
        data = yf.Ticker(ticker)
        ticker_marketCap = data.info['marketCap']
        print(ticker + ' Market Cap: $'+ str(ticker_marketCap) + " most recently")

    # Perform and show regression for given ticker, time, etc
    def show_stockreg(ticker: str, time_period: str, future_days: str, model_type: str) -> None:
        data,x_train,x_test = Stock.get_data(ticker,time_period,future_days)
        if model_type == "exp":
            mdl = Analysis.get_exp_linreg(x_train,data,x_test)
            Analysis.plot_regmdl(x_train,data,x_test,mdl,model_type)
        else: 
            mdl = Analysis.get_mult_linreg(x_train,data,model_type,x_test)
            Analysis.plot_regmdl(x_train,data,x_test,mdl,model_type)

    # Compare Two Plots with Polynomial Regression
    def compare_two_plot(ticker1: str, ticker2: str, time_period: str, future_days: str, order: int = 1) -> None:
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
    def plot_all_mdls(ticker: str, time_period: str, future_days: str, max_order: int = 3) -> None:
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
    def get_bitcoin_price(start_date: str,end_date: str = today) -> npt.NDArray:
        FRED_API_KEY = '8da06254f69eb0b6a6a0517042bb43f4'
        fred = Fred(api_key=FRED_API_KEY)
        data = np.array(fred.get_series('CBBTCUSD',start_date+' 00:00:00',end_date+' 00:00:00'))
        if Analysis.checknan(data,"BTC") == True:
            data = Analysis.rmvnan(data)
            print("removed nan")
        index = np.array(np.linspace(0,len(data),len(data)))
        return data, index

class Govt:
    # Get federal funds interest rate from FRED data
    def get_DFF(start_date: str,end_date: str = today) -> npt.NDArray:
        FRED_API_KEY = '8da06254f69eb0b6a6a0517042bb43f4'
        fred = Fred(api_key=FRED_API_KEY)
        data = fred.get_series('DFF',start_date+' 00:00:00',end_date+' 00:00:00').to_numpy()
        if Analysis.checknan(data,"DFF") == True:
            data = Analysis.rmvnan(data)
            print("removed nan")
        index22 = np.linspace(0,len(data),len(data))
        return data, index22

# Perform the analysis:
# stock1_pct_change = Stock.get_pct("BRK-A","10y")
# stock2_pct_change = Stock.get_pct("SBUX","10y")

# Analysis.show_correlaition(stock1_pct_change,stock2_pct_change)
# dff,time = Govt.get_DFF("01-01-2010")
# plt.plot(time,dff)
# plt.show()

# Stock.plot_all_mdls("GPRO","2mo","5",4)