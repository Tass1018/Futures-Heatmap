import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
#import kalman as k

file_dir='//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/project'

#import data
raw_data=pd.read_excel(file_dir+'DataDownload.xlsm',sheet_name='USDS',index_col='Dates')
raw_data=raw_data.dropna()
#generate spreads and flys
pillars=[i for i in raw_data.iloc[:,0:16].columns]
start_dt=input('Start Date For Charting/Analysis:  ')
raw_data=raw_data[pd.to_datetime(raw_data.index) >= pd.to_datetime(start_dt)]

def gen_spds(pillars):
    result=[]
    for i in pillars:
        for j in pillars:
            if j>i: result.append((i,j))
    return result

def gen_flys(pillars):
    result=[]
    for i in pillars:
        for j in pillars:
            for k in pillars:
                if k>j and j>i: result.append((i,j,k))
    return result

def check_hedge(x,outrights,spreads,flys):
    hrlist=outrights #outrights=pillars column names=[1,2,3,...,30]
    if 'Y' in str(x).upper(): #x= '1Y*2Y*3Y'
        if '*' in str(x):
            hrlist.extend(spreads)
            hrlist.extend(flys)
            return hrlist
        else: #'1Y2Y' ?? why not extend flys
            hrlist.extend(spreads)
            return hrlist
    else: #x=5*10 2*5*10?? 
        if '*' in str(x):
            hrlist.extend(spreads) #why spreads only?
            return hrlist       

def create_history(data,structure):
    """appends structure to data (where data exists to create structure in first place)"""
    NUDGE=1e-6 #small number to stop zero values for calculation later
    st_name='*'.join([str(i) for i in structure])
    if len(structure)==3: #create data frame for every possible arrangment of yrs rates
        wing1,belly,wing2=structure[0],structure[1],structure[2]
        s=2*data.loc[:,belly]-data.loc[:,wing1]-data.loc[:,wing2]+NUDGE
        data[st_name]=s
    if len(structure)==2:
        short_spd,long_spd=structure[0],structure[1]
        data[st_name]=data.loc[:,structure[1]]-data.loc[:,structure[0]]+NUDGE

def strat(input_string):
        if 'Y' in str(input_string).upper():
            return str(input_string).upper()
        else:
            return int(input_string)

def Chart_Structure(input_string,data): #calculations of HR of spreads and flys
    if '*' in str(input_string):
        split=input_string.split('*')
        if len(split)==2:
            result=(data.loc[:,strat(split[1])]-data.loc[:,strat(split[0])])
        if len(split)==3:
           result=(2*data.loc[:,strat(split[1])]-data.loc[:,strat(split[0])]-data.loc[:,strat(split[2])]) 
    elif isinstance(input_string,tuple):
        st_name='*'.join([str(i) for i in input_string])
        result=data.loc[:,st_name]
    else:
        result=data.loc[:,strat(input_string)]
    return result

spreads=gen_spds(pillars)
flys=gen_flys(pillars)

for fly in flys:
    print(fly)
    create_history(raw_data,fly)

for spd in spreads:
    create_history(raw_data,spd)

while True:
    x=input('Chart What?  ')
    y=input('Chart something else at same time? type "fb" for optimal  ')
    graph_data=Chart_Structure(x,raw_data) #hedge rate of x
    
    if y:
        if y.lower()!='fb':
            graph_data_2=Chart_Structure(y,raw_data)
        else:
            hedges=check_hedge(x,pillars,spreads,flys) #[1,2,...,(2,3),(2,4),..]
            adf_best=0
            for i in hedges:
                temp_data=Chart_Structure(i,raw_data) #hr value of i
                #mean,_,stat_bool,adf_res_adf,e,_=k.Kalman_Pair(df_hist[fly],df_hist[str(outright)],stationarity_test=True)
                reg = LinearRegression().fit(np.expand_dims(temp_data,-1), graph_data) #plot hr of x against hedge rate of i, y=bx+c
                res_values=graph_data-temp_data*np.squeeze(reg.coef_[0],axis=-1)-np.squeeze(reg.intercept_,axis=-1)
                res_adf=abs(adfuller(res_values, maxlag=3, regression='ct', autolag='AIC', store=False, regresults=False)[0]) #why maxlag is 3
                if res_adf>=adf_best: #you want to find the linear relationship that is most time non-stationary?
                    adf_best=res_adf
                    opt_hedge=i
            graph_data_2=Chart_Structure(opt_hedge,raw_data)
            print('Optimal Hedge: ',opt_hedge)
    """plot chart"""
    if y: fig,(ax1,ax2,ax3)=plt.subplots(3)
    else: fig,ax1=plt.subplots()
    ax1.plot(graph_data,label=str(x))
    ax1.legend(loc="upper right")

    if y:
        #ax2=ax.twinx()
        if y=='fb': ax2.plot(graph_data_2,label=str(opt_hedge))
        else:  ax2.plot(graph_data_2,label=str(y))
        ax2.legend(loc="upper right")
        # chart of residuals
        reg = LinearRegression().fit(np.expand_dims(graph_data_2,-1), graph_data)
        res_values=graph_data-graph_data_2*np.squeeze(reg.coef_[0],axis=-1)
        res_adf=adfuller(res_values, maxlag=3, regression='ct', autolag='AIC', store=False, regresults=False)
        ax3.plot(res_values,label='Trading residual. Beta:{:.2f}'.format(reg.coef_[0]))
        ax3.axhline(y=np.squeeze(reg.intercept_,axis=-1), color='r', linestyle='-')
        ax3.axhline(y=np.squeeze(reg.intercept_,axis=-1)+np.std(res_values)*2, color='b', linestyle='-')
        ax3.axhline(y=np.squeeze(reg.intercept_,axis=-1)-np.std(res_values)*2, color='b', linestyle='-')
        ax3.legend(loc="upper right")
        
    plt.legend()
    plt.show()


