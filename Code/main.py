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