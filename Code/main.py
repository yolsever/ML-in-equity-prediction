import numpy as np
import pandas as pd

df = pd.read_csv("..//"+"Data//"+"initial_dataset.csv", engine='python')

df = pd.get_dummies(df, columns=["gsector", "ggroup","gind", "gsector", "gsubind", "gvkey","tic", "gvkey"])
print(df.describe)
df = df.drop("iid",axis=1)
