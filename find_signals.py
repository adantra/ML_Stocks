
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import indicator
import pandas.io.data as web
import datetime
import pandasImpl
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.sandbox.regression.kernridgeregress_class import plt_closeall
import uuid
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def get_signals(stock,intervals):
    ibuy = 1;isell=2
    for inte in intervals:
        #print inte
        ii0=inte[0];ii1=inte[1]
        if stock['Close'].iloc[ii0]<stock['Close'].iloc[ii1]:
            stock['signal'].iloc[ii0] = ibuy
            stock['signal'].iloc[ii1] = isell
        else:
            stock['signal'].iloc[ii1] = ibuy
            stock['signal'].iloc[ii0] = isell


def find_ideal_trade_line2(inteval,stock,p):
    intervals = []
    stock['Det'] = 0
    done = False
    i0 = interval[0]
    ilast = interval[1]
    slope = (stock['Close'].iloc[ilast]-stock['Close'].iloc[i0])/(ilast-i0+1)
    b = stock['Close'].iloc[i0] - slope
    line = slope*np.arange(i0,ilast+1) - b
    threshold = 0.1*abs(stock['Close'].iloc[ilast]-stock['Close'].iloc[i0])
    #print line.shape
    #print stock['Close'].shape
    #for i in range(i0,ilast+1):
    stock['Det'].iloc[i0:ilast+1] = stock['Close'].iloc[i0:ilast+1] -line

    diff_max = -1000;diff_min = 1000
    i_max=0;j_max=0
    for i in np.arange(i0,ilast-2*p+1):
        for j in np.arange(i+p,ilast+1):
            diff = (np.abs(stock['Det'].iloc[j]-stock['Det'].iloc[i]))
            #print i,j,diff,diff_max
            cond1 = (i-i0>p)
            cond2 = (j-i>p)
            cond3 = (ilast-j>p)
            cond =cond1 and cond2 and cond3
            cond_diff = diff>0.05*stock['Close'].mean()
            slope2 = (stock['Close'].iloc[j]-stock['Close'].iloc[i])/(j-i+1)
            slope1 = (stock['Close'].iloc[ilast]-stock['Close'].iloc[i0])/(ilast-i0+1)
            cond4 = slope2*slope1<0
            cond = cond and cond4
            #print diff,0.1*stock['Close'].mean()
            #cond = cond and cond_diff
            if (diff > diff_max) and cond :
                i_max = i; j_max=j
                diff_max= diff
    #print i_max,j_max,diff,diff_max
    #if not(np.sign(stock['Det'].iloc[j_max]) == np.sign(stock['Det'].iloc[i_max])):
    slope2 = (stock['Close'].iloc[j_max]-stock['Close'].iloc[i_max])/(j_max-i_max+1)
    slope1 = (stock['Close'].iloc[ilast]-stock['Close'].iloc[i0])/(ilast-i0+1)
    cond1 = (i_max-i0>p)
    cond2 = (j_max-i_max>p)
    cond3 = (ilast-j_max>p)
    cond4 = slope2*slope1<0
    if cond1 and cond2 and cond3 and cond4:
        intervals.append((i0,i_max))
        intervals.append((i_max,j_max))
        intervals.append((j_max,ilast))
    elif cond2 and cond3:
        intervals.append((i0,j_max))
        intervals.append((j_max,ilast))
    elif cond1 and cond3:
        intervals.append((i0,i_max))
        intervals.append((i_max,ilast))
    elif cond1 and cond2:
        intervals.append((i0,i_max))
        intervals.append((i_max,ilast))


    if len(intervals)==0:
        intervals.append((i0,ilast))
    #print diff_max,intervals
    ii = 0
    #for inte in intervals:
    ##    ii0=inte[0];ii1=inte[1]
    #    if stock['Close'].iloc[ii0]<stock['Close'].iloc[ii1]:
    #        stock['signal'].iloc[ii0] = ibuy
    #        stock['signal'].iloc[ii1] = isell
    #    else:
    #        stock['signal'].iloc[ii1] = ibuy
    #        stock['signal'].iloc[ii0] = isell

    return intervals


ibuy=1
isell=2
start = datetime.datetime(2014,1,1)
end = datetime.datetime.today()
pcln = web.DataReader("pcln",'yahoo',start,end)
stock = pcln
stock['signal']=0
p = 10
i0=0;ilast=len(stock)-1
interval = (i0,ilast)
intervals = []
intervals.append(interval)
inter = []
for i in range(5):
    print 50*'/'
    print i
    inter = []
    for interval in intervals:
        #print interval
        if (interval[1]-interval[0]+1>2*p):
            inter = inter+find_ideal_trade_line2(interval,stock,p)
        #inter.append(inter)
    intervals = inter
    print 50*'\\'
#print inter
    get_signals(stock,intervals)
plt.plot(stock.index,stock['Close'])
plt.plot(stock[stock['signal']==ibuy].index,stock[stock['signal']==ibuy]['Close'],marker='o')
    #plt.plot(stock.index,stock['Close'])
plt.plot(stock[stock['signal']==isell].index,stock[stock['signal']==isell]['Close'],marker='o')
plt.show()
