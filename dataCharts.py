# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:34:09 2016

@author: GEOL05
"""

import numpy as np # linear algebra
import pandas as pd # data processing

data = pd.read_csv(r"C:\Users\GEOL05\Desktop\ComputerVisionChallenge\output\example.csv")
age_data = data.loc[(data['age'] <100) & (data['age'] > 0)]
data_by_age = age_data.groupby(['age']).size()
data_by_age.plot.bar(title='Age Histogram WIKI-IMDB', color = 'b', figsize=(18, 12))

less_than_18 = age_data.loc[age_data['age'] < 18]
print("Less than 18:", less_than_18['age'].count())

above_18 = age_data.loc[age_data['age'] >= 18]
print("Above 18:", above_18['age'].count())

less_than_30 = age_data.loc[age_data['age'] < 30]
print("Less than 30:", less_than_30['age'].count())

above_30 = age_data.loc[age_data['age'] >= 30]
print("Above 30:", above_30['age'].count())

d = {'<thres.':{'18': 1299, '30':27231},'>thres.':{'18': 59109, '30':33177}}

df = pd.DataFrame(d)
df.plot.bar(stacked=True, title="WIKI-IMDB");

d = {'<thres.':{'18': 598, '30':2268},'>thres.':{'18': 3515, '30':1845}}
df = pd.DataFrame(d)
df.plot.bar(stacked=True, title="LaP");
