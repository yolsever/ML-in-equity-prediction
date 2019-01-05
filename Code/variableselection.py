## variables selection methods here
## Was thinking of using CV Lasso and PCA, open to other suggestions

import numpy as np
import pandas as pd

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

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV(cv=5)

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]
    
    
plt.title(
    "Features selected using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()


