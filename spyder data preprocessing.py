#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:01:10 2024

@author: ayjeeg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# lets import dataset

df = pd.read_csv(r'/Users/ayjeeg/Downloads/FSDS prakash/Week 7/Data.csv')
df

# lets split the dataset based on dependent and independent variable

x = df.iloc[:, :-1].values # note: x should always be independent variable

y = df.iloc[:, 3].values # note: y should always be dependent variable

# we can fill the missing values in x through scikit package

from sklearn.impute import SimpleImputer
df = SimpleImputer() 
# we can use this:(within brackets) 
# missing_values=np.nan, strategy="median" 
# we can't use mode instead we use most_frequent
# for hyperparameter tuning

# lets impute the missing values with SimpleImputer

df = df.fit(x[ : , 1:3])
x[ : , 1:3] = df.transform(x[ : , 1:3])

# lets encode catagorical data and create a dummy variable 

from sklearn.preprocessing import LabelEncoder

# for x

label_x = LabelEncoder()
label_x.fit_transform(x[:,0])
x[:,0] = label_x.fit_transform(x[:,0])

# for y

label_y = LabelEncoder()
label_y.fit_transform(y)
y = label_y.fit_transform(y)

# lets split the data for training and testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# random stat keeps the split data constant
# without random stat the split data randomly split for 
# multiple execution of code and cause inaccuracy














