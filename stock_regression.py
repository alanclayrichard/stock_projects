from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

# input tickers to analyze and yfinance formatted time period (10y 2mo etc)
ticker1 = "MSFT"
ticker2 = "TSLA"
time_period = '10y'

# get data for tickerts
get_data1= yf.Ticker(ticker1)
get_data2 = yf.Ticker(ticker2)
ticker1_data = get_data1.history(period=time_period)
ticker2_data = get_data2.history(period=time_period)
# data will be closing prices for this model
close_prices1 = ticker1_data.iloc[:,3].to_numpy()
close_prices2 = ticker2_data.iloc[:,3].to_numpy()
# create an index to regress onto and plot and then create a prediction index to model future behavior 100 days into the future
index = np.linspace(0,len(close_prices1),len(close_prices1))
pred_index = np.linspace(0,len(close_prices1)+100,len(close_prices1)+100)

# # get recent prices
# tick1_text = ticker1 + " is at " + str(close_prices1[-1]) + " most recently"
# tick2_text = ticker2 + " is at " + str(close_prices2[-1]) + " most recently"
# print(tick1_text)
# print(tick2_text)

# # market cap
# ticker1_marketCap = get_data1.info['marketCap']
# ticker2_marketCap = get_data2.info['marketCap']
# print(ticker1 + ' Market Cap: '+ str(ticker1_marketCap))
# print(ticker2 + ' Market Cap: '+ str(ticker2_marketCap))

# plot share price
plt.figure(figsize=[14,6])
plt.plot(index,close_prices1)
plt.plot(index,close_prices2)
plt.xticks(size=4,rotation=45)

# Linear Model for 1
xhat1 = np.average(index)
yhat1 = np.average(close_prices1)
beta11 = np.sum((close_prices1-yhat1)*(index-xhat1))/np.sum((index-xhat1)**2)
beta01 = yhat1 - beta11*xhat1
pred1 = beta11*pred_index + beta01
# print(beta11)
# print(beta01)
plt.plot(pred_index,pred1)

#linear model for 2
xhat2 = np.average(index)
yhat2 = np.average(close_prices2)
beta12 = np.sum((close_prices2-yhat2)*(index-xhat2))/np.sum((index-xhat2)**2)
beta02 = yhat2 - beta12*xhat2
pred2 = beta12*pred_index + beta02
print(beta12)
print(beta02)
plt.plot(pred_index,pred2)

# quadtratic model for 1
sum_matrix1 = np.array([np.sum(index**4), np.sum(index**3), np.sum(index**2), np.sum(index**3), np.sum(index**2), np.sum(index), np.sum(index**2), np.sum(index), len(index)])
sum_matrix1 = np.reshape(sum_matrix1,[3,3])
# print(sum_matrix)
sum_solutions1 = (np.array([np.sum(np.multiply(index**2,close_prices1)),np.sum(np.multiply(index,close_prices1)),np.sum(close_prices1)]))
sum_solutions1 = np.reshape(sum_solutions1,[3,1])
print(sum_solutions1)
betas1 = np.dot(np.linalg.inv(sum_matrix1),sum_solutions1)
print(betas1)
quad_prediction1 = betas1[2] + betas1[1]*pred_index + betas1[0]*(pred_index**2)
plt.plot(pred_index,quad_prediction1)

# quadratic model for 2
sum_matrix2 = np.array([np.sum(index**4), np.sum(index**3), np.sum(index**2), np.sum(index**3), np.sum(index**2), np.sum(index), np.sum(index**2), np.sum(index), len(index)])
sum_matrix2 = np.reshape(sum_matrix2,[3,3])
# print(sum_matrix2)
sum_solutions2 = (np.array([np.sum(np.multiply(index**2,close_prices2)),np.sum(np.multiply(index,close_prices2)),np.sum(close_prices2)]))
sum_solutions2 = np.reshape(sum_solutions2,[3,1])
print(sum_solutions2)
betas2 = np.dot(np.linalg.inv(sum_matrix2),sum_solutions2)
print(betas2)
quad_prediction2 = betas2[2] + betas2[1]*pred_index + betas2[0]*(pred_index**2)
plt.plot(pred_index,quad_prediction2)

# format plot
plt.title(ticker1 + " and " + ticker2 + " Analysis")
plt.ylabel("share price")
plt.xlabel("previous " + time_period)
plt.legend([ticker1,ticker2,ticker1+" linear model",ticker2+" linear model",ticker1+" quadratic model",ticker2+" quadratic model"])
plt.show()
