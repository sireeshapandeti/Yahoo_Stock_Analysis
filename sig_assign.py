
##Importing Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
import datetime
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

## raw data to csv format
yahoo_data = pd.read_csv('/Users/sireeshapandeti/Documents/PythonProgs/yahoo_midterm.csv')


## Basic data analysis
## Displays first 5 rows of the data
print(yahoo_data.head(5))

## Cleaning the data
yahoo_data = pd.read_csv('/Users/sireeshapandeti/Documents/PythonProgs/yahoo_midterm.csv', parse_dates = True, index_col = 0)

## Displays first 5 rows of the data
print(yahoo_data.head(5))

## Displays last 5 rows of the data
print(yahoo_data.tail(5))

## Displays basic information of the data
print(yahoo_data.info())

## Displays statistics of the data
print(yahoo_data.describe())


##Plot1 :: Line Plot
yahoo_data.High.plot(kind = 'line', color = 'b', label = 'High', linewidth=1, alpha = .85, linestyle = '-')
yahoo_data.Low.plot(color = 'y', label = 'Low', linewidth=1, alpha = .85, linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('Time[Days]')              
plt.ylabel('Value')
plt.title('Line Plot(Time Vs Value)')            
plt.show()


 ##Plot2 :: Histogram
yahoo_data.Volume.plot(kind='hist', color='blue', figsize=(5,5), alpha=.7)
plt.xlabel('Time[Days]')
plt.ylabel('Value')
plt.title('Stock_Volume according to Time',color='blue',size='14')
plt.legend(loc='upper right')
plt.show()

## Dropping NULL's
yahoo_data.dropna(inplace = True)

## Open, High, Low and Close values
import plotly.plotly as py
import plotly.graph_objs as go

trace = go.Ohlc(x=yahoo_data['Date'],
                open=yahoo_data['Open'],
                high=yahoo_data['High'],
                low=yahoo_data['Low'],
                close=yahoo_data['Close'])
data = [trace]

##Plot3 :: Graphical view of Open, High, Low and Close
py.plot(data, filename = 'basic-line', auto_open=True, xlabel= 'Date', ylabel='Price')

## Stock Returns 
returns = pd.DataFrame()
returns['Returns'] = yahoo_data['Close'].pct_change()
print(returns.head())

## Least single day gains
#print("Minimum Returns day :", returns.idxmin())

## Biggest single day gains
#print("Maximum Returns day :", returns.idxmax())

### High STD , The price varies a lot-------means the stock is risky. Low STD, the price is steady
#print("Standard Deviation ::", returns.std())


#Plot4 :: Seaborn Dist Plot
sns.distplot(returns.ix['2018-01-01':'2018-10-25']['Returns'], color='red', bins=50)
plt.show()


#Plot5 :: Plot the 30 day average against the close price 
plt.figure(figsize=(12,4))
yahoo_data['Close'].ix['2013-10-30':'2018-10-25'].rolling(window=30).mean().plot(label='30 day moving averagge')
yahoo_data['Close'].ix['2013-10-30':'2018-10-25'].plot(label='Close')
plt.legend()
plt.show()


############## Linear Regression ##############
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#print(yahoo_data.columns)

X = yahoo_data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = yahoo_data['Adj Close']

## Train and Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

lm = LinearRegression()

lm.fit(X_train, y_train)

## Coefficients
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
print(cdf)

## Predictions
predictions = lm.predict(X_test)
print(predictions)

#Plot6 :: Scatter Plot
plt.scatter(y_test, predictions)
plt.show()

#Plot7 :: Distribution Plot
sns.distplot(y_test - predictions)
plt.show()

## Accuracy
print("Accuracy of Linear Regerssion Model:",lm.score(X_test,y_test))



















