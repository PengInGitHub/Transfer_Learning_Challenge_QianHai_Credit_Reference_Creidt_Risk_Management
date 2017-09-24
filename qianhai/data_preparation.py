#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 18:22:57 2017

@author: pengchengliu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import xgboost as xgb
import random
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#######################################
#             change data             #
#######################################
train = pd.read_csv('A_train.csv')

#following three lines run at the very first time only
#train.rename(columns={'flag':'y'}, inplace=True)
#train.rename(columns={'no':'uid'}, inplace=True)
#train.to_csv('A_train.csv',index=None)

#rate of pos cases and ratio of pos to neg
d=train
len(d[(d['y']==1)])/(len(d[(d['y']==0)]))#15:100

train_xy, test_xy = train_test_split(train, test_size = 0.2)
#train_xy 32,000*158
#test_xy 8,000*158
train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]

train_xy[['uid']].to_csv('uid.csv',index=None)
train_xy[['uid','y']].to_csv('train_y.csv',index=None)
test_xy[['uid','y']].to_csv('test_y.csv',index=None)

train_xy.iloc[:, train.columns != 'y'].to_csv('train_x.csv',index=None)
test_xy.iloc[:, train.columns != 'y'].to_csv('test_x.csv',index=None)



uid = pd.read_csv('uid.csv')
train_x = pd.read_csv('train_x.csv')
train_y = pd.read_csv('train_y.csv')
test_x= pd.read_csv('test_x.csv')
test_y=pd.read_csv('test_y.csv')

feature_type = pd.read_csv('feature_type.csv')#see below

#######################################
#             data type               #
#######################################
#define numeric variables:
#all columns have more than 10 distinct values are seen as numeric varianles
unique_value_stat = pd.Series.sort_values(train.apply(pd.Series.nunique))
numerical_feature = list((unique_value_stat[unique_value_stat>10].index).values)
categorical_feature = list((unique_value_stat[unique_value_stat<=10].index).values)
numerical_feature.remove('uid')
categorical_feature.remove('y')

my_df = pd.DataFrame(numerical_feature+categorical_feature)
my_df['feature_type']='numeric'
my_df.loc[156:,'feature_type']='category'
my_df.columns=["feature", "type"]
my_df.to_csv('feature_type.csv', index=False)





   

