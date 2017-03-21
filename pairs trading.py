import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from  datetime import datetime, timedelta
import statsmodels.api as sm
import pytz


def initialize(context):

    context.stocks=symbols('IYJ','IYW','IYH','IYZ','IDU','IYF',
                                  'IYE','IYK','IYM','IYC')
    context.index=symbol('SPY')
    
    set_commission(commission.PerTrade(cost=0))
    context.spreads=[]

    context.window_length=181
    context.zscore_length=181
    context.zscore_invested=[]
    
    context.invested=False
    
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
    
def before_trading_start(context, data):
    
    stocks_his=data.history(context.stocks,'price',           
                            context.window_length,
                            '1d').iloc[0:context.window_length-1].dropna()
    
    index_his=data.history(context.index,'price',           
                            context.window_length,
                            '1d').iloc[0:context.window_length-1].dropna()
    
    context.stocks_his=stocks_his
    context.index_his=index_his
    
    beta=PCRegression(context,context.index_his,context.stocks_his)
    zscore=compute_zscore(context,context.spreads)
    
    context.beta=beta
    context.zscore=zscore
         
def handle_data(context,data):
    
    record(zscore=context.zscore,position=context.portfolio.positions_value)
    
    
def PCRegression(context,Y, X):

    pca = PCA()
    A = scale(X)
    B = scale(Y)
    
    a1 = np.ones((A.shape[0],A.shape[1] + 1))
    a1[:, 1:] = A
    A = a1
    
    newA = pca.fit_transform(A)[:,:-1]
    a2 = np.ones((newA.shape[0], newA.shape[1] + 1))
    a2[:, 1:] = newA
    newA = a2
    
    regr1 = sm.OLS(B, A).fit()
    beta=regr1.params   
    
    regr2 = sm.OLS(B, newA).fit()
    pc_beta=regr2.params
    
    
    spread=B[-1]-np.dot(newA[-1,0:7],pc_beta[0:7])
    context.spreads.append(spread)
    
    return beta

def compute_zscore(context,spreads): 

    if len(spreads)<context.zscore_length:
        return

    spread_wind=spreads[-(context.zscore_length-1):]
    zscore=(spreads[-1]-np.mean(spread_wind))/np.std(spread_wind)
    context.zscore_length+=1
    
    return zscore

def my_rebalance(context,data):
    
    place_order(context,data,context.zscore,context.beta)
    context.window_length+=1
    


def place_order(context,data,zscore,beta):

    if zscore== None:
        return
    
    stocks_price=data.current(context.stocks,'price')
    index_price=data.current(context.index,'price') 
    a = np.ones((1,stocks_price.count() + 1))
    a[:,1:] = np.array(stocks_price)
    stocks_price_ = a
    x_=np.dot(stocks_price_,beta)
    
    
    y_value=context.portfolio.portfolio_value*index_price\
           /(index_price+x_)
    x_value=context.portfolio.portfolio_value-y_value
    x_weight=x_value*beta[1:]*stocks_price/np.dot(beta[1:],stocks_price)
    
    if zscore>=1.2 and not context.invested:
        for i in range(len(context.stocks)):
                order_target_value(context.stocks[i],4*x_weight[i])   
        order_target_value(context.index,-4*y_value)
        context.invested=True
        context.zscore_invested=zscore
        
    elif zscore<=-1.2 and not context.invested:
        for i in range(len(context.stocks)):
                order_target_value(context.stocks[i],-4*x_weight[i])
        order_target_value(context.index,4*y_value)
        context.invested=True
        context.zscore_invested=zscore        
        
    elif np.abs(zscore)<(np.abs(context.zscore_invested)-1)and context.invested:    
         for stock in context.stocks:
            order_target_value(stock,0)
         order_target_value(context.index,0)
         context.invested=False

    

    

