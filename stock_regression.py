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

# Linear Model
def get_simple_linreg(X,Y,X_prediction):
    x = X
    y = Y
    xhat = np.average(x)
    yhat = np.average(y)
    beta1 = np.sum((y-yhat)*(x-xhat))/np.sum((x-xhat)**2)
    beta0 = yhat - beta1*xhat
    Y_prediction = beta1*X_prediction + beta0
    return Y_prediction

# quadtratic model for 1
def get_quad_linreg(X,Y,X_prediction):
    sum_matrix = np.array([np.sum(X**4), np.sum(X**3), np.sum(X**2), np.sum(X**3), np.sum(X**2), np.sum(X), np.sum(X**2), np.sum(X), len(X)])
    sum_matrix = np.reshape(sum_matrix,[3,3])
    # print(sum_matrix)
    sum_solutions = (np.array([np.sum(np.multiply(X**2,Y)),np.sum(np.multiply(X,Y)),np.sum(Y)]))
    sum_solutions = np.reshape(sum_solutions,[3,1])
    betas = np.dot(np.linalg.inv(sum_matrix),sum_solutions)
    quad_mdl = betas[2] + betas[1]*X_prediction + betas[0]*(X_prediction**2)
    return quad_mdl

# # exponential model for 1
def get_exp_linreg(X,Y,X_prediction):
    log_prices = np.log(Y[1:len(Y)])
    log_index = X[1:len(X)]
    x_loghat1 = np.average(log_index)
    y_loghat1 = np.average(log_prices)
    log_beta1 = np.sum((log_prices-y_loghat1)*(log_index-x_loghat1))/np.sum((log_index-x_loghat1)**2)
    log_beta0 = y_loghat1 - log_beta1*x_loghat1
    log_pred = np.exp(log_beta1*(X_prediction[1:len(X_prediction)]) + log_beta0)
    return log_pred

# # Specify tickers and parameters to analyze
ticker1 = "AAPL"
ticker2 = "MSFT"
time_period1 = "3y"
time_period2 = "3y"
future_days1 = "100"
future_days2 = "100"

# # Get the data for the tickers defined
# get_current_price(ticker1)
# get_current_price(ticker2)
# get_mkt_cap(ticker1)
# get_mkt_cap(ticker2)
close_prices1,index1,pred_index1 = get_data(ticker1,time_period1,future_days1)
close_prices2,index2,pred_index2 = get_data(ticker2,time_period2,future_days2)

# # Make the models (simple linear regression, quadratic linear regression, exponential linear regression)
lin_model1 = get_simple_linreg(index1,close_prices1,pred_index1)
lin_model2 = get_simple_linreg(index2,close_prices2,pred_index2)
quad_mdl1 = get_quad_linreg(index1,close_prices1,pred_index1)
quad_mdl2 = get_quad_linreg(index2,close_prices2,pred_index2)
exp_mdl1 = get_exp_linreg(index1,close_prices1,pred_index1)
exp_mdl2 = get_exp_linreg(index2,close_prices2,pred_index2)

# # Make the plots to compare
plt.figure(figsize=[14,6])
plt.plot(index1,close_prices1,zorder=7)
plt.plot(index2,close_prices2,zorder=8)
plt.plot(pred_index1,lin_model1,"--")
plt.plot(pred_index2,lin_model2,"--")
plt.plot(pred_index1,quad_mdl1,"--")
plt.plot(pred_index2,quad_mdl2,"--")
plt.plot(pred_index1[1:len(pred_index1)],exp_mdl1,"--")
plt.plot(pred_index2[1:len(pred_index2)],exp_mdl2,"--")
plt.title(ticker1 + " and " + ticker2 + " Analysis")
plt.ylabel("share price")
plt.xlabel("previous " + time_period1 + "("+ticker1+")" + " or "+ time_period2+ "("+ticker2+")")
plt.legend([ticker1,ticker2,ticker1+" linear model",ticker2+" linear model",ticker1+" quadratic model",ticker2+" quadratic model",ticker1+" exponential model",ticker2+" exponential model"])
plt.show()
