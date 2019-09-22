import warnings
warnings.filterwarnings('ignore')
import requests
from pandas_datareader import data
import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
%matplotlib inline
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta
from dateutil import parser


API_KEY = "ReplaceWithYourKey"



def MakeDataFrameFromResponse(response):
    values = pd.DataFrame()
    values['date'] = pd.Series()
    values['value'] = pd.Series()
    _date_ = []
    _values_ = []
    for i in (response.json()['observations']):
        _date_.append(i['date'])
        _values_.append(i['value'])
    values['date'] = _date_
    values['value'] = _values_
    return values


def JoinDataOnThreeMonthsPrior(x, LookbackPeriod):
    '''
    The intention here was that if we can see an indicator that moves prior to the stock, we 
    can use that indicators movement to predict the stock's movement.    
    '''
    try:
        dt = parser.parse(x)
    except Exception as e:
        dt = x
    output = dt + relativedelta(months =+ LookbackPeriod)
    return output



def CollectIndicators(ListOfIndicators, LookbackPeriod):
    '''
    Author:Steel Ricciotti
    Date:12-30-2018
    Description:This function concatenates indicators into a single dataframe.
    '''
    existingdataframe = pd.DataFrame(columns=['date','value'])
    existingdataframe.index = existingdataframe.date
    oldsuffix = 'value'
    for counter,i in enumerate(ListOfIndicators):
        existingdataframe = existingdataframe.reindex()
        existingdataframe['date']  = existingdataframe.index
        try:
            if counter > 0:
                existingdataframe['date']  = existingdataframe.index       
            try:
                _response_ = requests.get("https://api.stlouisfed.org/fred/series/observations?series_id=" + i + "&api_key="+ API_KEY + "&file_type=json")
                newdataframe = (MakeDataFrameFromResponse(_response_))
                newdataframe  = newdataframe.dropna()
                newdataframe.columns = ['date', i]
                newdataframe.index = newdataframe.date.apply(_DateFormatter_)
                newdataframe['date'] = newdataframe['date'].apply(lambda x: JoinDataOnThreeMonthsPrior(x, LookbackPeriod))
                try:
                    newdataframe = newdataframe[pd.to_numeric(newdataframe[i], errors='coerce').notnull()]
                except Exception as e:
                    print(e)
                '''The first indicator actually populates a dataframe'''
                if counter == 0:
                    existingdataframe = newdataframe
                    existingdataframe.columns = ['date', i]
                elif existingdataframe.index.size > newdataframe.index.size:
                    existingdataframe = newdataframe.set_index('date').join(existingdataframe.set_index('date'),lsuffix =i, rsuffix = oldsuffix)
                else:
                    existingdataframe = existingdataframe.join(newdataframe.set_index('date'),lsuffix =oldsuffix, rsuffix = i)
                oldsuffix = i 
            except Exception as e:
                print(i, " was not found")
                econ_indic.remove(i)
        except Exception as e:
            econ_indic.remove(i)
            print(i, " had an issue")
            
    
    existingdataframe.index = existingdataframe['date']
    return existingdataframe.dropna()
                     

def makeint(input):
    __ = input.split("-")
    return str(__[0]) + str(__[1]) + str(__[2])
    
def _DateFormatter_(input):
    return input.split(" ")[0]

def MergeStocksAndIndicators(stock, indicator):
    #indicator.index = indicator.date.apply(_DateFormatter_)
    #stock['Date'] = stock.index.strftime("%Y-%m-%d")
    stock['Date'] = stock.index
    #stock.index = stock.index.strftime("%Y-%m-%d")
    stock.index = stock.index
    #indicator.index = indicator.index.strftime("%Y-%m-%d")
    indicator = indicator.loc[indicator.index > str(min(stock['Date']))]
    stock[stock_] = stock['Adj Close']
    stock = stock[['Date',stock_]]
    if stock['Date'].size > indicator.index.size:
        dataset = indicator.join(stock)
    else:
        dataset = stock.join(indicator)
    return  dataset.reindex()



def process(econ_indic,stock_, LookbackPeriod,start_date, end_date):
    stockname = stock_
    stock = data.DataReader(stock_, 'yahoo', start_date, end_date)
    stock = stock.resample('MS').first()
    highscore = 0
    highestscore = 0
    for lookback in range(0,6):
        try:
            highscore = 0
            indicators = CollectIndicators(econ_indic, lookback)
            indicators.date = indicators.index            
            indicators.index = indicators.date.apply(lambda x: JoinDataOnThreeMonthsPrior(x, LookbackPeriod))
            combination = MergeStocksAndIndicators(stock, indicators)
            combination = combination.dropna()
            try:
                econ_indic.remove(stock_)
            except Exception as e:
                pass
            X = combination[stock_].values.reshape(-1, 1)
            '''
            If doing multi-variate linear regression reshape X
            '''
            values = combination.columns.tolist()
            values.remove(stock_)
            try:
                values.remove('Date')
            except Exception as e:
                pass
            try:
                values.remove('date')
            except Exception as e:
                pass
            X = combination[values].values.reshape(-1,len(values))
            y = combination[stock_].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            newscore = reg.score(X, y)
            if newscore > highscore:
                highscore = newscore
                highestscore = lookback
            else:
                highscore = highscore
                highestscore = highestscore
        except Exception as e:
            print(e)
            
            
    indicators = CollectIndicators(econ_indic, highestscore)
    indicators.date = indicators.index            
    indicators.index = indicators.date.apply(lambda x: JoinDataOnThreeMonthsPrior(x, highestscore))
    combination = MergeStocksAndIndicators(stock, indicators)
    combination = combination.dropna()
    try:
        econ_indic.remove(stock_)
    except Exception as e:
        pass
    X = combination[stock_].values.reshape(-1, 1)
    '''
    If doing multi-variate linear regression reshape X
    '''
    values = combination.columns.tolist()
    values.remove(stock_)
    try:
        values.remove('Date')
    except Exception as e:
        pass
    try:
        values.remove('date')
    except Exception as e:
        pass
    X = combination[values].values.reshape(-1,len(values))
    y = combination[stock_].values.reshape(-1, 1)
    __date = "2018-12-01"
    data__ = combination.loc[combination.index == __date]
    combination_ = data__
    Xpredict = combination_[values].values.reshape(-1,len(values))    
    reg = LinearRegression().fit(X, y)
    _score_ = reg.score(X, y)
    print("Prediction for " + str(stockname) + " on " + str(__date) + " is $" +  str(round(float(reg.predict(Xpredict)[0][0]),2)) + " based on " + str(econ_indic[0]) + " with a score of " + str(_score_))
    plt.figure(figsize=(15,5))
    plt.xlabel = econ_indic[0]
    plt.ylabel = stock
    plt.label = econ_indic[0]
    data__[econ_indic[0]] = data__[econ_indic[0]].astype(float).astype(int)
    for i in econ_indic:
        data__[i] = data__[i].astype(float).astype(int)
    data__.index = pd.to_datetime(data__.index)
    X = combination_[values].values.reshape(-1,len(values))
    data__['Prediction'] = reg.predict(X)
    econ_indi = econ_indic
    econ_indi.append(stock_)
    econ_indi.append('Prediction')
    data__[econ_indi].plot(figsize=(20, 15), subplots=True); 
    return data__


start_date = '2010-01-01'
end_date = '2019-12-31'    
stock_ = "TOOL.CN"
econ_indic = ["TOTALSEC","MSACSR", "PERMIT","MRTSSM44611USN","BUSINV","CEFDFSA066MSFRBPHI"]
data__ = process(econ_indic,stock_,2,startdate, enddate)
