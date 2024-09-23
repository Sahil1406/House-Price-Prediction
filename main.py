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

print(house_price_data)