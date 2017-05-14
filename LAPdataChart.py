# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:34:09 2016

@author: GEOL05
"""

import numpy as np # linear algebra
import pandas as pd # data processing

data = pd.read_csv(r"C:\Users\GEOL05\Desktop\ComputerVisionChallenge\output\train_gt.csv")
age_data = data.round({'mean': 0})
print (age_data.head())
age_data = age_data.loc[(age_data['mean'] <100) & (age_data['mean'] > 0)]
data_by_age = age_data.groupby(['mean']).size()
data_by_age.plot.bar(title='Age Histogram LaP', color = 'r', figsize=(18, 12))

#count data

less_than_18 = age_data.loc[age_data['mean'] < 18]
print("Less than 18:", less_than_18['mean'].count())

above_18 = age_data.loc[age_data['mean'] >= 18]
print("Above 18:", above_18['mean'].count())

less_than_30 = age_data.loc[age_data['mean'] < 30]
print("Less than 30:", less_than_30['mean'].count())

above_30 = age_data.loc[age_data['mean'] >= 30]
print("Above 30:", above_30['mean'].count())