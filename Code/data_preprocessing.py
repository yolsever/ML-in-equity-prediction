import numpy as np
import pandas as pd

df = pd.read_csv("..//"+"Data//"+"initial_dataset.csv", engine='python')
df = pd.get_dummies(df, columns=["gsector", "ggroup","gind", "gsector", "gsubind", "gvkey","tic", "gvkey"])
print(df.describe)
df = df.drop("iid",axis=1)


## Creates new dependant variables for direction of stock movement (we want to forecast the price direction)
## We can use the same code for other time periods of course
dependantclose = df['prccd'].diff()
newdf = pd.DataFrame(dependantclose)
newdf.loc[newdf['prccd'] < 0, "direction"] = 0 
newdf.loc[newdf['prccd'] > 0, "direction"] = 1
df = pd.concat([df, newdf['direction']], axis=1, sort=False)


## This code splits the dataset above into datasets sorted by company and saves them
stocklist = pd.unique(df['tic'])
for i in stocklist: 
    df1 = df[df['tic'] == i]
    # df1.to_csv(i)

## Loading company level Data (access all company strings in the list "stocklist")
## Also important to drop the first row! I'll explain why

df1 = pd.read_csv("..//"+"Data//"+"Stockdata//"+"GOOGL")
df1.drop(df.index[0])


## ADDING TECHNICAL INDICATORS

from technicalindicators import *


def technical(df1):
    df = df1
    
    T = [5,15,30,60]
    for i in T:
        df = moving_average(df,i)
        df = exponential_moving_average(df,i)
        df = momentum(df,i)
        df = rate_of_change(df, i)
        df = average_true_range(df,i)
        df = vortex_indicator(df, i)
        df = relative_strength_index(df, i)
        df = money_flow_index(df, i)
        df = on_balance_volume(df, i)
        df = force_index(df, i)
        df = ease_of_movement(df, i)
        df = commodity_channel_index(df, i)
        df = coppock_curve(df, i)
        df = keltner_channel(df, i)
        df = donchian_channel(df, i)
        df = standard_deviation(df, i)
    
    J = [5,15,30]    
    for i in J:
        df = bollinger_bands(df, i)
        df = stochastic_oscillator_d(df, i)
        df = trix(df, i)
        df = accumulation_distribution(df, i)
        
    ## Time independant
    df = ppsr(df)
    df = stochastic_oscillator_k(df)
    df = mass_index(df)
    df = chaikin_oscillator(df)
    df = ultimate_oscillator(df)
    return df
  
    
    
#df = moving_average(df1,5)
df = technical(df1)