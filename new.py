from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from urllib import request

# input tickers
ticker1 = "MSFT"
ticker2 = "AAPL"

# get data
get_data1= yf.Ticker(ticker1)
get_data2 = yf.Ticker(ticker2)
ticker1_data = get_data1.history(period='10y')
ticker2_data = get_data2.history(period='10y')
tick1_text = ticker1 + " closed at " + str(ticker1_data.iloc[len(ticker1_data)-1,3]) + " most recently"
tick2_text = ticker2 + " closed at " + str(ticker2_data.iloc[len(ticker2_data)-1,3]) + " most recently"
print(tick1_text)
print(tick2_text)

# market cap
ticker1_marketCap = get_data1.info['marketCap']
ticker2_marketCap = get_data2.info['marketCap']
print(ticker1 + 'Market Cap: '+ str(ticker1_marketCap))
print(ticker2 + 'Market Cap: '+ str(ticker2_marketCap))

# plot 
plt.figure(figsize=[10,4])
plt.plot(ticker1_data.iloc[:,3])
plt.plot(ticker2_data.iloc[:,3])
plt.xticks(size=4,rotation=45)
plt.legend([ticker1,ticker2])
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