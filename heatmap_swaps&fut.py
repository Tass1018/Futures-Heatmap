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
raw_data_fut=pd.read_excel(file_dir+'USD_futures.xlsx',sheet_name='sofr_cm',index_col='Dates')
raw_data_fut=raw_data_fut.dropna()

df_fut=raw_data_fut.loc['1/01/2020':] # 1.1and1.2 value missing, starts from 1.3
df_fut=df_fut.iloc[:,:16]
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



def heatmap_df_single(df_swap, df_fut):
    swap = df_swap.iloc[:,1:6]
    
    sfr_spd = pd.DataFrame(columns=df_fut.columns, index = swap.columns)
    best_r2=0
    best_r2_res=[]
    betas=pd.DataFrame(columns=df_fut.columns, index = swap.columns)

    
    for row in range(sfr_spd.shape[0]):
        for col in range(sfr_spd.shape[1]):
            rowName=sfr_spd.index[row]
            colName=sfr_spd.columns[col]
            y = np.array(swap.loc[:,rowName]).reshape(-1, 1)
            y=y.reshape(-1, 1)
            
            x = np.array(df_fut.loc[:,colName]).reshape(-1, 1)
            x=x.reshape(-1, 1)
            
        
            
            #linear regression swap spread against future spread
            model=LinearRegression().fit(x,y)
            #residual errors
            beta=model.coef_[0]

            res = y-x*np.squeeze(model.coef_[0],axis=-1)-np.squeeze(model.intercept_,axis=-1)
            #squared sum of residual errors
#             SS_Residual = np.sum((res)**2)   
            r2=model.score(x,y)
            if r2 > best_r2:
                best_r2=r2
                best_r2_res=res
            sfr_spd.loc[rowName,colName]=r2
            betas.loc[rowName,colName]=beta[0]
    
                
    return sfr_spd,best_r2_res,betas

heatmap_single=heatmap_df_single(df_swap, df_fut)
df=heatmap_single[0]
best_r2_res=heatmap_single[1]
betas=heatmap_single[2]

df=df[df.columns].astype(float) 
df=df*100
fig, ax = plt.subplots(figsize=(11, 9))
# plot heatmap
sb.heatmap(df, cmap='Blues', vmin= 0.0, vmax=100,
        linewidth=0.3, annot=df, annot_kws={'va':'bottom'})
sb.heatmap(df,cmap='Blues', vmin= 0.0, vmax=100,annot=betas.round(2), annot_kws={'va':'top'},cbar=False)
title = 'R SQUARE HEATMAP OF\nSOFR FUTURES AGAINST SWAPS (R2 / Beta)'
plt.title(title, loc='left', fontsize=18)
plt.show()

