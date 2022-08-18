import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


file_dir='//SVRSG001RPS01.asia.corp.anz.com/huy11$/My Documents/heatmap/'

#import data
raw_data_swap=pd.read_excel(file_dir+'DataDownload.xlsm',sheet_name='USDS',index_col='Dates')
raw_data_swap=raw_data_swap.dropna()
raw_data_fut=pd.read_excel(file_dir+'usd_fut.xlsx',sheet_name='sofr_cm',index_col='Dates')
raw_data_fut=raw_data_fut.dropna()

df_fut=raw_data_fut.loc['1/01/2020':] # 1.1and1.2 value missing, starts from 1.3
df_fut=df_fut.iloc[:,4:20]
df_fut=df_fut.diff()*100
df_fut=df_fut.dropna()
df_fut['date'] = df_fut.index.tolist()

df_swap=raw_data_swap.diff()*100
df_swap.index = df_swap.index
df_swap=df_swap.dropna()
df_swap=df_swap.iloc[3:,:]

list2=df_swap.index.tolist()
l=[]
for day in list2:
    day=day.replace(hour=0)
    l.append(day)
df_swap['date'] = l
df_swap=df_swap.groupby('date').sum()

aligned_df=pd.merge(df_swap,df_fut,left_on='date',right_on='date')

df_swap=aligned_df.iloc[-90:,:122]
df_fut = aligned_df.iloc[-90:,123:]
df_fut['10Y_Swap']=df_swap.iloc[-90:,10]

################################################################################
def strat(input_string):
        if 'Y' in str(input_string).upper():
            return str(input_string).upper()
        else:
            return int(input_string)
        
def swap_spread(input_string, df_swap):
     if '*' in str(input_string):
        split=input_string.split('*')
        if len(split)==3:
            result=(2*df_swap.loc[:,strat(split[1])]-df_swap.loc[:,strat(split[0])]-df_swap.loc[:,strat(split[2])]) 
            return result.tolist()
        else:
            print("length is wrong!")
            return -1


def heatmap_df(df_fut, result):
    sfr_spd = pd.DataFrame(columns=df_fut.columns, index = df_fut.columns)
    
    best_r2=0
    best_r2_res=[]
    best_model_colname=''
    best_model_rowname=''
    # betas1=pd.DataFrame(columns=df_fut.columns, index = df_fut.columns)
    # betas2=pd.DataFrame(columns=df_fut.columns, index = df_fut.columns)
    y = result
    
    
    betas=pd.DataFrame(columns=df_fut.columns, index = df_fut.columns)
    
    for row in range(sfr_spd.shape[0]):
        for col in range(sfr_spd.shape[1]):
            rowName=sfr_spd.index[row]
            colName=sfr_spd.columns[col]
            if rowName!=colName and col < row:
                #calculate the future spread
                fut_pair_df= df_fut.loc[:, [colName, rowName]]
                
                x = fut_pair_df
                
                #linear regression swap spread against future spread
                model=LinearRegression().fit(x,y)
                #residual errors
                
                beta1=model.coef_[0]
                beta2=model.coef_[1]
                
                y_prediction =  model.predict(x)
                res=y-y_prediction
                # res = y-x*np.squeeze(model.coef_[0],axis=-1)-np.squeeze(model.intercept_,axis=-1)
                #squared sum of residual errors
                SS_Residual = np.sum((res)**2)   
                r2=model.score(x,y)
                if r2 > best_r2:
                    best_r2=r2
                    best_r2_res=res
                    best_model_colname=colName
                    best_model_rowname=rowName
                sfr_spd.loc[rowName,colName]=r2
                # betas1.loc[rowName,colName]=beta1.round(2)
                # betas2.loc[rowName,colName]=beta2.round(2)
                best_model_name=best_model_colname+'*'+best_model_rowname
                
                
                betas.loc[rowName,colName]=beta1.round(2).__str__()+ " / "+beta2.round(2).__str__()
                
    
    
                
    return sfr_spd,best_r2_res,best_model_name, betas



##########################
import plotly.express as px
import plotly.graph_objects as go
while True:
    x=input('Chart What?  (eg: 5*7, 20Y30Y*30Y10Y)')
    result=swap_spread(x, df_swap)
    heatmap=heatmap_df(df_fut, result)
    df=heatmap[0]
    best_r2_res=heatmap[1]
   
    res_cumsum=np.cumsum(best_r2_res,axis=0)
    best_model_name=heatmap[2]
    betas=heatmap[3]
    
    df=df[df.columns].astype(float) 
    fig, ax = plt.subplots(figsize=(33, 18))
    # plot heatmap
    sb.heatmap(df, cmap='Blues', vmin= 0.0, vmax=1,
            linewidth=0.5, annot=df, annot_kws={'va':'bottom', 'fontsize': 13, 'fontstyle': 'oblique', 'alpha': 1}, fmt=".1%")

    sb.heatmap(df,linewidth=0.5,cmap='Blues', vmin= 0.0, vmax=1,annot=betas, annot_kws={'va':'top', 'color':'red'},cbar=False, fmt='')
    

    title = 'R SQUARE HEATMAP OF\nSOFR FUTURES AGAINST FLYS (R2 / Beta1-lower axis / Beta2-side axis)'
    plt.title(title, loc='left', fontsize=18)
    plt.show()
    
    fig = px.line(best_r2_res, title="Residual Error"+x+" against "+best_model_name)
    fig1 = px.line(res_cumsum, title="Cumulative Residual Error"+x+" against "+best_model_name)
    fig.update_xaxes(
                title_text = "Date",
                title_font = {"size": 15},
                title_standoff = 25)

    fig.update_yaxes(
            title_text = "Residual Error Index",
            title_font = {"size": 15},
            title_standoff = 25)
    fig.show()
    fig1.show()