## variables selection methods here
## Was thinking of using CV Lasso and PCA, open to other suggestions

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# loading random company
df1 = pd.read_csv("..//"+"Data//"+"Stockdata//"+"AMAT.csv")
df1.drop(df1.index[0])

## Bunch of transformations also defining covariates and dependant variables
#df1 = pd.get_dummies(df1, columns=["gsector", "ggroup","gind", "gsector", "gsubind", "gvkey","tic", "gvkey", "spcsrc", "incorp", "idbflag"])
#df1 = df1.reset_index()
df1 = df1.drop(["capgn","spcseccd","incorp","spcsrc", "idbflag", "tic", "prccd", "prchd", "prcld", "prcod"],axis=1)
df1 = df1.dropna()


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

X, y = df1.drop(["direction"], axis=1), df1['direction']

reg = LassoCV(cv=20, random_state=0).fit(X, y)
reg.get_params
