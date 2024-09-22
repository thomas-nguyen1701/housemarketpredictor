import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("California_Houses.csv")

data.dropna(inplace=True)

from sklearn.model_selection import train_test_split

x = data.drop(['Median_House_Value'], axis=1) #training model
y = data['Median_House_Value']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


train_data = x_train.join(y_train)

plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

train_data ['Tot_Rooms'] = np.log(train_data['Tot_Rooms'] + 1)
train_data ['Tot_Bedrooms'] = np.log(train_data['Tot_Bedrooms'] + 1)
train_data ['Population'] = np.log(train_data['Population'] + 1)
train_data ['Households'] = np.log(train_data['Households'] + 1)

train_data.hist(figsize=(15,8))

train_data ['bedroom_ratio'] = train_data['Tot_Bedrooms'] / train_data['Tot_Rooms']
train_data ['households_rooms'] = train_data['Tot_Rooms'] / train_data['Households']
train_data ['bedroom_ratio'] = train_data['Tot_Bedrooms'] / train_data['Tot_Rooms']\
    
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")