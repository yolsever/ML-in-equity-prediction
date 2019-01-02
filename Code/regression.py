## Testing decision trees with skLearn!

import numpy as np
import pandas as pd

## Loading company level Data (access all company strings in the list "stocklist")
## Also important to drop the first row! I'll explain why

df1 = pd.read_csv("..//"+"Data//"+"Stockdata//"+"GOOGL")
df1.drop(df1.index[0])

## Bunch of transformations also defining covariates and dependant variables
df1 = pd.get_dummies(df1, columns=["gsector", "ggroup","gind", "gsector", "gsubind", "gvkey","tic", "gvkey", "spcsrc", "incorp", "idbflag"])
#df1 = df1.reset_index()
df1 = df1.drop(["capgn","spcseccd"],axis=1)
df1 = df1.dropna()
y = df1['direction']
X = df1.drop(["direction"], axis=1)

## Making test splits: Test size is how much of the data we use as training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

## Training the Data with decision trees and random forest
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l1')

classifier.fit(X_train, y_train)

## Making the actual prediciton
y_pred = classifier.predict(X_test)

## Some elementary diagnostics, pretty good results!
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
