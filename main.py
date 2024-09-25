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

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train, y_train = train_data.drop(['Median_House_Value'], axis=1), train_data['Median_House_Value']
x_train_s = scaler.fit_transform(x_train)

reg = LinearRegression()

reg.fit(x_train_s, y_train)
LinearRegression()

test_data = x_test.join(y_test)

test_data ['Tot_Rooms'] = np.log(test_data['Tot_Rooms'] + 1)
test_data ['Tot_Bedrooms'] = np.log(test_data['Tot_Bedrooms'] + 1)
test_data ['Population'] = np.log(test_data['Population'] + 1)
test_data ['Households'] = np.log(test_data['Households'] + 1)

test_data ['bedroom_ratio'] = test_data['Tot_Bedrooms'] / test_data['Tot_Rooms']
test_data ['households_rooms'] = test_data['Tot_Rooms'] / test_data['Households']

x_test,y_test = test_data.drop(['Median_House_Value'], axis = 1), test_data['Median_House_Value']
x_test_s = scaler.transform(x_test)
reg.score(x_test_s,y_test)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(x_train, y_train)
forest.score(x_test, y_test)

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [300, 50, 125],
    "max_features": [4, 8, 16],
    "min_samples_split": [2, 4]
}

grid_search = GridSearchCV(forest, param_grid, cv= 5, scoring= "neg_mean_squared_error", return_train_score=True)

grid_search.fit(x_train_s, y_train)



