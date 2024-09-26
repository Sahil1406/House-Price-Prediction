# Importing all the dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# importing all the dataset

house_price_data = sklearn.datasets.fetch_california_housing()

# print(house_price_data)

# Loading the data into a pandas dataset

dataset = pd.DataFrame(house_price_data.data, columns= house_price_data.feature_names)

dataset['price'] = house_price_data.target

# print(dataset.head())

# print(dataset.shape)

# print(dataset.isnull().sum())

# print(dataset.describe())

# UNDERSTANGING VARIOUS RELATIONSHIPS BETWEEN VARIOUS FEATURES IN THE DATASETS

correlation = dataset.corr()

# constructing a heatmap for the correlation

plt.figure(figsize=(20,20))
# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

# splitting the dataset into 2 variables which has price and the features

X = dataset.drop('price', axis=1)
Y = dataset['price']

# print(Y)
# print(X)

# splitting the data into training and testing parts

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)

# print(X.shape, X_test.shape, X_train.shape)

# TRAINING THE MODEL ON THE TRAINING SPLIT

model = XGBRegressor()

model.fit(X_train, Y_train)


# EVALUATIONS

# PREDICTING THE VALUE OF TRAINING SET

training_data_prediction = model.predict(X_train)

# print(training_data_prediction)

# R SQUARED ERROR

score1 = metrics.r2_score(Y_train, training_data_prediction)

# MEAN ABSOLUTE ERROR

score2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

# print('R Squared Error: ', score1)
# print('Mean Absolute Error: ', score2)


# PREDICTION ON TESTING DATA

testing_data_prediction = model.predict(X_test)

# R SQUARED ERROR

scoret1 = metrics.r2_score(Y_test, testing_data_prediction)

# MEAN ABSOLUTE ERROR

scoret2 = metrics.mean_absolute_error(Y_test, testing_data_prediction)

print('R Squared Error: ', scoret1)
print('Mean Absolute Error: ', scoret2)


# VISUALIZING THE PREDICTED VALUES VS ACTUAL VALUES

plt.scatter(Y_train, training_data_prediction)
plt.xlabel = 'Actual Price'
plt.ylabel = 'Predicted Price'
plt.title = 'Actual Price vs Prediction Price'
# plt.show()