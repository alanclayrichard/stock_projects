from typing import OrderedDict
from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from urllib import request
# from sklearn.linear_model import LinearRegression

# input tickers
ticker1 = "MSFT"
ticker2 = "TSLA"

# get data
get_data1= yf.Ticker(ticker1)
get_data2 = yf.Ticker(ticker2)
ticker1_data = get_data1.history(period='10y')
ticker2_data = get_data2.history(period='10y')
close_prices1 = ticker1_data.iloc[:,3].to_numpy()
close_prices2 = ticker2_data.iloc[:,3].to_numpy()
index = np.linspace(0,len(close_prices1),len(close_prices1))
# tick1_text = ticker1 + " is at " + str(close_prices1[-1]) + " most recently"
# tick2_text = ticker2 + " is at " + str(close_prices2[-1]) + " most recently"
# print(tick1_text)
# print(tick2_text)

# # market cap
# ticker1_marketCap = get_data1.info['marketCap']
# ticker2_marketCap = get_data2.info['marketCap']
# print(ticker1 + ' Market Cap: '+ str(ticker1_marketCap))
# print(ticker2 + ' Market Cap: '+ str(ticker2_marketCap))

# plot 
plt.figure(figsize=[10,4])
plt.plot(index,close_prices1)
plt.plot(index,close_prices2)
plt.xticks(size=4,rotation=45)

# Linear Model for 1
xhat1 = np.average(index)
yhat1 = np.average(close_prices1)
beta11 = np.sum((close_prices1-yhat1)*(index-xhat1))/np.sum((index-xhat1)**2)
beta01 = yhat1 - beta11*xhat1
pred1 = beta11*index + beta01
print(beta11)
print(beta01)
plt.plot(index,pred1)

#linear model for 2
xhat2 = np.average(index)
yhat2 = np.average(close_prices2)
beta12 = np.sum((close_prices2-yhat2)*(index-xhat2))/np.sum((index-xhat2)**2)
beta02 = yhat2 - beta12*xhat2
pred2 = beta12*index + beta02
print(beta12)
print(beta02)
plt.plot(index,pred2)
plt.title(ticker1 + " and " + ticker2 + " Analysis")
plt.ylabel("share price")
plt.legend([ticker1,ticker2,ticker1+" linear model",ticker2+" linear model"])
plt.show()


# # Treasury Data
# data = pd.read_csv("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates?fields=record_date,security_desc,avg_interest_rate_amt&filter=record_date:gte:2021-12-01&format=csv")
# dates = data['record_date']
# description = data['security_desc']
# rates = data['avg_interest_rate_amt']
# print(len(dates))
# print(len(description))
# print(len(rates))
# print(dates,description,rates)

# data.plot()
# # # data = np.array(data,[:,2])
# # # print(data)
# # plt.plot(data[0],data[1])
# # plt.show()
# plt.show()