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
#from astropy._erfa.erfa_generator import args

warnings.filterwarnings("ignore")


def find_long_trad(istart,iend,n,stock):
    diff_max = -1000
    ibuy=0;isell=0
    for i1 in range(istart,iend-n,1):
        for i2 in range(istart+n,iend,1):
            diff = stock['Close'].iloc[i2]-stock['Close'].iloc[i1]
            if diff>diff_max:
                ibuy = i1;isell=i2
                #print ibuy,isell
                #print stock.index[i1],stock.index[i2]
                diff_max=diff
                #print diff_max
    
    if (diff_max/stock['Close'].iloc[i1]*100)>1:
        stock['GT0'].iloc[ibuy] = 1
        stock['GT0'].iloc[isell] = 2
    
    return ibuy,isell,diff_max

def find_long_trad2(istart,iend,n,stock):
    diff_max = -1000
    for i1 in range(istart,iend-n,1):
        for i2 in range(istart+n,iend,1):
            diff = stock['Close'].iloc[i2]-stock['Close'].iloc[i1]
            if diff>diff_max:
                ibuy = i1;isell=i2
                #print ibuy,isell
                #print stock.index[i1],stock.index[i2]
                diff_max=diff
                #print diff_max
    
    if (diff_max/stock['Close'].iloc[i1]*100)>1:
        stock['GT0'].iloc[ibuy:isell] = 1
        #stock['GT0'].iloc[isell] = 2
    
    return ibuy,isell,diff_max

def find_short_trad(istart,iend,n,stock):
    diff_max = -1000
    for i1 in range(istart,iend-n,1):
        for i2 in range(istart+n,iend,1):
            diff = -stock['Close'].iloc[i2]+stock['Close'].iloc[i1]
            if diff>diff_max:
                isell = i1;ibuy=i2
                #print ibuy,isell
                #print stock.index[i1],stock.index[i2]
                diff_max=diff
                #print diff_max
    
    if (diff_max/stock['Close'].iloc[i1]*100)>1:
        stock['GT0'].iloc[ibuy] = 1
        stock['GT0'].iloc[isell] = 2
    
    return ibuy,isell,diff_max

def get_signal(ntype,stock,*argv,**kargv):
    if ntype == 1:
        stock['GAIN_30'] = stock['Close'].\
                          diff(-period)/stock['Close']*100
        stock['GT0'] = 1*(stock['GAIN_30']>2)+2*(stock['GAIN_30']<-2)
    elif ntype == 2:
        period = argv[0] #20
        istart=0
        n= argv[1]
        iend = istart+period
        stock['GT0'] = 0
        while iend<len(stock):
            ibuy,isell,diff_max = find_long_trad(istart,iend,n,stock)
            istart = isell
            iend = isell+period
    elif ntype == 3:   
        period = argv[0] #20
        istart=0
        n= argv[1]
        iend = istart+period
        stock['GT0'] = 0 
        while iend<len(stock):
            ibuy,isell,diff_max = find_short_trad(istart,iend,n,stock)
            istart = ibuy
            iend = ibuy+period
        #    print istart,iend,diff_max,len(stock)
        #print 20*'*'+'  done  '+20*'*' 
    elif ntype == 4:
        period = argv[0] #20
        istart=0
        n= argv[1]
        iend = istart+period
        stock['GT0'] = 0
        while iend<len(stock):
            ibuy,isell,diff_max = find_long_trad2(istart,iend,n,stock)
            istart = isell
            iend = isell+period
    
    
    

def plot_decision_regions(X, y, classifier, 
                    test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]                               
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

def trade_strategy(n,period,stock):
    ### Buy at first signal, hold for period, then buy at next signal
    initial_money = 10000
    cash = initial_money
    i=0
    n_shares = 0
    ibuy=1
    for ii in stock.index:
        row = stock.iloc[i]  
        if row['Date'].year == 2015:   
            if row['ada_pre']==1 and cash>row['Close']:
               ### Buy
                n_shares = int(cash/row['Close'])
                cash = cash - n_shares*row['Close']
                print 'Date',row['Date']
                print 'CASH',cash,n_shares
                print 'BUY %3i SHARES at % 5.2f' % (n_shares,row["Close"])
                print 50*'$'
                ibuy =1 
            elif n_shares>0 and np.mod(ibuy,period) ==0:
                cash += n_shares*row['Close']
                print 'IBUY',ibuy
                print 'Date',row['Date']
                print 'CASH',cash,n_shares
                print 'SELL %3i SHARES at % 5.2f' % (n_shares,row["Close"])
                print 50*'@'
                n_shares = 0
            elif n_shares>0:
                ibuy+=1
            
        i+=1
    print 'Number of shares = ',n_shares
    print 'Cash = ', cash
    money = cash + n_shares*row['Close']
    return money/initial_money -1

def trade_strategy_short(n,period,stock):
    ### Buy at first signal, hold for period, then buy at next signal
    initial_money = 10000
    cash = initial_money
    i=0
    n_shares = 0
    ibuy=1
    for ii in stock.index:
        row = stock.iloc[i]  
        if row['Date'].year == 2009:   
            if row['ada_pre']==1 and cash>row['Close']:
               ### Buy
                n_shares = int(cash/row['Close'])
                cash = cash - n_shares*row['Close']
                print 'Date',row['Date']
                print 'CASH',cash,n_shares
                print 'BUY %3i SHARES at % 5.2f' % (n_shares,row["Close"])
                print 50*'$'
                ibuy =1 
            elif n_shares>0 and np.mod(ibuy,period) ==0:
                cash += n_shares*row['Close']
                print 'IBUY',ibuy
                print 'Date',row['Date']
                print 'CASH',cash,n_shares
                print 'SELL %3i SHARES at % 5.2f' % (n_shares,row["Close"])
                print 50*'@'
                n_shares = 0
            elif n_shares>0:
                ibuy+=1
            
        i+=1
    print 'Number of shares = ',n_shares
    print 'Cash = ', cash
    money = cash + n_shares*row['Close']
    return money/initial_money -1
          
        
    

def get_stocks():
    tickers = ['aapl','goog','pcln','bidu','isrg']
    return tickers

def get_technicals(stock):
    #stock=pandasImpl.MACD(stock,26,12)
    #stock=pandasImpl.MACD(stock,40,20)
    #stock=pandasImpl.MACD(stock,50,25)
    #stock=pandasImpl.MACD(stock,80,40)
    stock=pandasImpl.Chaikin(stock)
    stock=pandasImpl.ULTOSC(stock)
    n_min = 10;n_max=120;n_step=10
    for n in range(n_min,n_max,n_step):
        stock=pandasImpl.MACD(stock,n,n/2)
        #stock=pandasImpl.CCI(stock,n)
        stock=pandasImpl.MA(stock, n)
        stock=pandasImpl.BBANDS(stock,n)        
        stock=pandasImpl.ROC(stock, n)
        stock=pandasImpl.ATR(stock, n)
        stock=pandasImpl.RSI(stock, n)
        stock=pandasImpl.FORCE(stock, n)
        stock=pandasImpl.OBV(stock, n)
        stock=pandasImpl.STDDEV(stock, n)
        stock=pandasImpl.EOM(stock, n)
        #stock=pandasImpl.MFI(stock, n)
       
        #stock=pandasImpl.TRIX(stock, n)
    
    #print stock.head()
    return stock

def normalize_stock(stock):
    stock = pandasImpl.NormAdjClose(stock)
    #stock = pandasImpl.Norm_by_mean(stock)
    #make index a couter
    #stock['Date']=stock.index
    stock.index = range(len(stock))
    #print stock.head()
    return stock
def func(a,*args):
    if args:
        print a, args

def plot(stock1,ticker,*argv,**kargs):
    markersize=10
    stock2 = stock1[stock1['LR_Pre']==1]
    stock4 = stock1[stock1['LR_Pre']==2]
    plt.plot(stock2['Date'],stock2['Close'],marker='^',\
             linestyle='None',color='g',markersize=markersize) 
    plt.plot(stock4['Date'],stock4['Close'],marker='v',\
             linestyle='None',color='r',markersize=markersize)
    plt.plot(stock1['Date'],stock1['Close'])
    title = ticker + 'Period = '+ str(argv[0])
    #if argv:
     #   print argv
        #title = ticker + ' ' +str(args)
    
    plt.title(title)
    plt.savefig(ticker+str(argv[0])+'_lr')
    plt.close()
    
    stock3 = stock1[stock1['ada_pre']==1]
    stock5 = stock1[stock1['ada_pre']==2]
    plt.plot(stock5['Date'],stock5['Close'],marker='v',\
             linestyle='None',color='r',markersize=markersize)
    plt.plot(stock3['Date'],stock3['Close'],marker='^',\
             linestyle='None',color='g',markersize=markersize)
    
    plt.plot(stock1['Date'],stock1['Close'])
    title = ticker + 'Period = '+ str(argv[0])
    #if argv:
     #   print argv
        #title = ticker + ' ' +str(args)
    
    plt.title(title)
    plt.savefig(ticker+str(argv[0])+'_ada')
    plt.close()
    #plt.show()



def logistic(tickers,start,end,period):
    unique_folder = str(uuid.uuid4())
    print unique_folder
    os.mkdir(unique_folder)
    os.chdir(unique_folder)
    f = open('info.txt', 'w')
    f.write('Period = %i \n' % period)
    f.write('Start Date = '+str(start)+'\n')
    f.write('End Date = ' + str(end)+'\n')
    f.write('Strategy = ' + '\n')
    f.write('Fraction for training = %0.2f \n' % frac_training)
    f.write('Strategy = %i \n' % n_strategy)
    f.write('Minumum Delta = %i \n' % minimum_delta)
    #max_depth = 1
    f.write('Max Depth (adaBoost) = %i \n' % max_depth)
    #n_estimators = 1000
    f.write('n_estimators (adaBoost) = %i \n' % n_estimators)
    
    total_error=0
    total_earnings=0
    random_earnings=0
    ### run trough all the tickers
    for ticker in tickers:
        print 50*'#'
        ### Read data from the web
        stock = web.DataReader(ticker,'yahoo',start,end)
        f.write(50* '\/')
        f.write('\n')
        f.write('Ticker = ' + ticker+'\n')
        f.write('Total data points %3i \n' %len(stock))
        dates = stock.index
        ### Extract column names before 
        stock_headers = stock.columns.values.tolist()
        #Normalize stocks and get technical indicators
        stock = normalize_stock(stock)
        stock = get_technicals(stock)
        # Copy after technical have been computed
        stock_orig=stock.copy()
        
        
        #plot(stock_orig)
        
        stock_orig=stock_orig.dropna()  #Drop NaN due to technical calculation
        ### Get feature list
        stock_tech_head = stock.columns.values.tolist()
        features = list(set(stock_tech_head)-set(stock_headers))
        ### Define target 
        
        
        
        get_signal(n_strategy, stock,period,minimum_delta)
        
        
        #stock['GAIN_30'] = stock['Close'].\
        #              diff(-period)/stock['Close']*100
        #stock['GT0'] = 1*(stock['GAIN_30']>2)+2*(stock['GAIN_30']<-2)
        #stock['GT0'] = 2*(stock['GAIN_30']<-5)
        stock=stock.dropna()
        ### Select specific features we are going to be working with
        used_features = features[:len(features)]
        ### Extract Feature and Target values
        X = stock[used_features].as_matrix() 
        y = stock['GT0'].as_matrix()
        ### Split in train and test sets
        X_train,X_test,y_train,y_test = \
                    train_test_split(X,y,test_size=0.2, random_state=0)
        X_train = X[:len(X)*frac_training]
        y_train = y[:len(X)*frac_training]
        X_test = X[len(X)*frac_training:]
        y_test = y[len(X)*frac_training:]
        f.write('Number of points for training = %i \n' %len(X_train))
        f.write('Number of points for testing = %i \n' %len(X_test))
        ### Scale features 
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        X_std = sc.transform(X)
        #X_test = X_test_std
        frac_1=sum(y_train)/(len(y_train)*1.0)
        
        ##### Decision Tree with AdaBoost
        tree = DecisionTreeClassifier(criterion='entropy',\
                                        max_depth=1,random_state=0,class_weight='auto')
        
        ada = AdaBoostClassifier(base_estimator=tree,n_estimators=n_estimators, \
                                 learning_rate=0.1, random_state=0)
        
        tree.fit(X_train_std, y_train)
        y_train_tree_pred = tree.predict(X_train_std)
        y_test_tree_pred = tree.predict(X_test_std)
        tree_train = accuracy_score(y_train, y_train_tree_pred)
        tree_test = accuracy_score(y_test, y_test_tree_pred)
        print('Decision tree train/test accuracies %.3f/%.3f' \
              % (tree_train, tree_test))
        f.write('Decision tree train/test accuracies %.3f/%.3f \n' \
              % (tree_train, tree_test))
        ### AdaBoost
        ada = ada.fit(X_train_std, y_train)
        y_train_tree_pred = ada.predict(X_train_std)
        y_test_tree_pred = ada.predict(X_test_std)
        ada_train = accuracy_score(y_train, y_train_tree_pred)
        ada_test = accuracy_score(y_test, y_test_tree_pred) 
        print('AdaBoost train/test accuracies %.3f/%.3f \n'\
              % (ada_train, ada_test))
        f.write('AdaBoost train/test accuracies %.3f/%.3f \n'\
              % (ada_train, ada_test))
        #lr = LogisticRegression(C=1, random_state=0,class_weight='auto')
        lr = RandomForestClassifier(max_depth=5, n_estimators=10, \
                                    max_features=5, class_weight='auto')
        
        #lr = LogisticRegression(C=1,class_weight={0:1-frac_1,1:frac_1}, random_state=0)
        ## random binary to compare with random trading 
        rb = np.random.randint(2,size=len(stock))
        stock['Random']=rb
        #
        ### Perdorm logistic regression fit and store predictions
        #
        lr.fit(X_train_std, y_train)
        y_train_predictions = lr.predict(X_train_std)
        y_test_predictions = lr.predict(X_test_std)
        lr_train = accuracy_score(y_train, y_train_predictions)
        lr_test = accuracy_score(y_test, y_test_predictions)
        print('Logistic Regression train/test accuracies %.3f/%.3f \n'\
              % (lr_train, lr_test))
        f.write('Logistic Regression train/test accuracies %.3f/%.3f \n'\
              % (lr_train, lr_test))
        ## predctions for whole history
        y_predict_all = lr.predict(X_std)
        #prob_all = lr.predict_proba(X_std)
        stock['Prediction'] = y_predict_all
        #stock['Probability'] = prob_all[:,1]
        ### Add predictons and Probabilities to the original set
        
        stock_pos = stock[stock['Prediction']==1]
        
        print 'Ticker = ', ticker
        print 'Logistics'
        print(' samples: %d' % len(y_test))
        samples = 1.0*len(y_test)
        mis = (y_test != y_test_predictions).sum()
        print('Misclassified samples: %d' % (y_test != y_test_predictions).sum())
        print( 'Error', mis/samples)
        #print 'Earnings LR',stock_pos['GAIN_30'].sum()/(len(X)*period)*200
        #print stock_pos[['GAIN_30','Probability','Adj Close']].tail()
        #print stock['GAIN_30','Prediction','Probability'].tail()
        print 10*'@#$$'
        print 'fraction',frac_1
        #total_earnings +=stock_pos['GAIN_30'].sum()/(len(X)*period)*200
        total_error += mis/samples
        #random_earnings += stock[stock['Random']==1]['GAIN_30'].sum()/(len(X)*period)*200
    
        ### Print recommendation with full data
        X = stock_orig[used_features].as_matrix() 
        sc = StandardScaler()
        sc.fit(X)
        X_std = sc.transform(X)
        y_predict = lr.predict(X_std)
        y_predict_ada = ada.predict(X_std)
        y_proba = lr.predict_proba(X_std)
        stock_orig['ada_pre'] = y_predict_ada
        stock_orig['LR_Pre'] = y_predict
        stock_orig['LR_Pro'] = y_proba[:,1] 
        stock_orig['Date']=dates[len(dates)-len(stock_orig):]
    
        print 80*'/'
        print "Current recommendation for ", '***',ticker, '***'
        #print stock_orig[['Close','Prediction','Probability','Date']].tail()
        print stock_orig[['Date','Close','LR_Pre','ada_pre']].iloc[-50:]
        stock_orig.index = stock_orig['Date']
        stock_test = stock_orig.iloc[int(len(X)*frac_training):]
        stock_pos_ada = stock_test[stock_orig['ada_pre']==1] #.iloc[int(len(X)*0.8):]
        stock_pos_lr = stock_test[stock_orig['LR_Pre']==1] 
        #earn = trade_strategy(1,period,stock_orig)
        #print 'EARNINGS',earn
        plot(stock_test,ticker,period)
        
    f.close()
            
    print 15*'%&*'
    #print 'Total Earnings = ',total_earnings/len(tickers)
    print 'Total Error = ',total_error/len(tickers)
    print 'Random Earnings = ',random_earnings/len(tickers)
    


tickers  = ['aapl','goog','pcln','isrg','cop',\
           'amzn','sbux','xom','nflx','azo','faz','vti',\
           'mmm','bio'] #faz, bidu

#tickers  = ['mmm','bio']

#tickers = ['ba']

#tickers  = ['goog','nflx']
max_depth = 1
n_estimators = 100
n_strategy = 3
minimum_delta = 5
start = datetime.datetime(2005,1,1)
func(1,start)
#end = datetime.datetime(2009,5,19)
#end = datetime.datetime(2016,5,20)
end = datetime.date.today()
period = 30
frac_training = 0.95

logistic(tickers, start, end, period)   
#neural_net(tickers, start, end, period)        
    
    