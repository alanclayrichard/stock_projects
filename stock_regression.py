from matplotlib import pyplot as plt
import yfinance as yf
import numpy as np


# Get data for tickers
def get_data(ticker,time_period,days_future):
    data = yf.Ticker(ticker)
    time_data = data.history(period=time_period)
    # data will be closing prices for this model
    close_prices = time_data.iloc[:,3].to_numpy()
    # create an index to regress onto and plot and then create a prediction index to model future behavior 100 days into the future
    index = np.linspace(0,len(close_prices),len(close_prices))
    pred_index = np.linspace(0,len(close_prices)+int(days_future),len(close_prices)+int(days_future))
    return close_prices,index,pred_index

# Get current prices
def get_current_price(ticker):
    data = yf.Ticker(ticker)
    time_data = data.history(period="10d")
    close_prices = time_data.iloc[:,3].to_numpy()
    tick_text = ticker + " is at $" + str(close_prices[-1]) + " per share most recently"
    print(tick_text)

# market cap
def get_mkt_cap(ticker):
    data = yf.Ticker(ticker)
    ticker_marketCap = data.info['marketCap']
    print(ticker + ' Market Cap: $'+ str(ticker_marketCap) + " most recently")


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

# # exponential model for 1
def get_exp_linreg(X,Y,X_prediction):
    log_prices = np.log(Y[1:len(Y)])
    log_index = X[1:len(X)]
    x_loghat1 = np.average(log_index)
    y_loghat1 = np.average(log_prices)
    log_beta1 = np.sum((log_prices-y_loghat1)*(log_index-x_loghat1))/np.sum((log_index-x_loghat1)**2)
    log_beta0 = y_loghat1 - log_beta1*x_loghat1
    exp_mdl = np.exp(log_beta1*(X_prediction[1:len(X_prediction)]) + log_beta0)
    return exp_mdl

# # Specify tickers and parameters to analyze
ticker1 = "AAPL"
ticker2 = "SPY"
time_period1 = "2y"
time_period2 = "2y"
future_days1 = "50"
future_days2 = "50"

# # Get the data for the tickers defined
# get_current_price(ticker1)
# get_current_price(ticker2)
# get_mkt_cap(ticker1)
# get_mkt_cap(ticker2)
close_prices1,index1,pred_index1 = get_data(ticker1,time_period1,future_days1)
close_prices2,index2,pred_index2 = get_data(ticker2,time_period2,future_days2)

poly_linreg_mdl = get_poly_linreg(index1,close_prices1,2,pred_index1)

# # Make the models (simple linear regression, quadratic linear regression, exponential linear regression)
lin_mdl1 = get_poly_linreg(index1,close_prices1,1,pred_index1)
lin_mdl2 = get_poly_linreg(index2,close_prices2,1,pred_index2)
quad_mdl1 = get_poly_linreg(index1,close_prices1,2,pred_index1)
quad_mdl2 = get_poly_linreg(index2,close_prices2,2,pred_index2)
exp_mdl1 = get_exp_linreg(index1,close_prices1,pred_index1)
exp_mdl2 = get_exp_linreg(index2,close_prices2,pred_index2)

# # # Make the plots to compare
plt.figure(figsize=[14,6])
plt.plot(index1,close_prices1,zorder=7)
plt.plot(index2,close_prices2,zorder=8)
plt.plot(pred_index1,lin_mdl1,"--",linewidth=.75)
plt.plot(pred_index2,lin_mdl2,"--",linewidth=.75)
plt.plot(pred_index1,quad_mdl1,"--",linewidth=.75)
plt.plot(pred_index2,quad_mdl2,"--",linewidth=.75)
plt.plot(pred_index1[1:len(pred_index1)],exp_mdl1,"--",linewidth=.75)
plt.plot(pred_index2[1:len(pred_index2)],exp_mdl2,"--",linewidth=.75)
plt.title(ticker1 + " and " + ticker2 + " Analysis")
plt.ylabel("share price")
plt.xlabel("previous " + time_period1 + "("+ticker1+")" + " or "+ time_period2+ "("+ticker2+")")
plt.legend([ticker1,ticker2,ticker1+" linear model",ticker2+" linear model",ticker1+" quadratic model",ticker2+" quadratic model",ticker1+" exponential model",ticker2+" exponential model"])
plt.show()
