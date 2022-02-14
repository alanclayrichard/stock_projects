from cmath import exp
from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np


# Get Data for Tickers
def get_data(ticker,time_period,days_future):
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

# Polynomial Regression
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

# # Get the data for the tickers defined
# print_current_price(ticker1)
# print_current_price(ticker2)
# print_mkt_cap(ticker1)
# print_mkt_cap(ticker2)

# compare_two_plot("AAPL","MSFT","10y","100")
# plt.show()

plot_all_mdls("AAPL","1mo","0",8)
plt.show()
