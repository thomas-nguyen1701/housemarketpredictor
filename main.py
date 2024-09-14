import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("California_Houses.csv")

data.dropna(inplace=True)

data.info()

from sklearn.model_selection import train_test_split

x = data.drop(['Median_House_Value']) #training model
y = data['Median_House_Value']
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


train_data = x_train.join(y_train)
train_data.hist()