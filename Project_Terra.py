# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:25:47 2019

@author: Pepecito_
    Pre_Processing Notes:
        As of now the NaN values will be replaced with the mean but I will
        need to research more later when I get this thing working
    Statistics to Research:
        koi_ingress,
        koi_longp,
        
        
        
        
        
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('Cumulative_KOI_2019.07.26_10.58.04.csv', error_bad_lines=False, index_col=0)
dropped_cols = [
        'kepid',
        'kepoi_name',
        'kepler_name', 
        'koi_vet_date',
        'koi_score',
        'koi_fpflag_nt',
        'koi_fpflag_ss',
        'koi_fpflag_co',
        'koi_fpflag_ec',
        'koi_pdisposition', 
        'koi_comment',
        'koi_disp_prov',
        'koi_parm_prov',
        'koi_sparprov',
        'koi_datalink_dvs',
        'koi_datalink_dvr',
        'koi_limbdark_mod',
        'koi_trans_mod',
        'koi_model_dof',
        'koi_model_chisq',
        'koi_tce_delivname',
        'koi_tce_plnt_num',
        'koi_limbdark_mod',
        'koi_fittype',
        'koi_vet_stat',
        'koi_longp',
        'koi_ingress',
        'koi_trans_mod',
        'koi_model_dof',
        'koi_model_chisq',
        'koi_sage',
        ]
df=df.drop(dropped_cols,axis=1)

#First I need to drop uncessary/problematic columns.
#Then split remaining data into two categories by vetting status: DONE and 
#We will be using the DONE candidates to train and test.
#Drop the vetting status.
candidates = df.loc[(df['koi_disposition'] == 'CANDIDATE') & (df['koi_disposition']  == 'NOT DISPOSITIONED') ]
df = df.loc[(df['koi_disposition'] != 'CANDIDATE') & (df['koi_disposition']  != 'NOT DISPOSITIONED') ]

myDict = {'CONFIRMED':1,'FALSE POSITIVE':-1}
df=df.replace(myDict)
print(df)


df = df.fillna(df.mean())
df.isna().any()

X = df.drop('koi_disposition', axis=1)
Y = df['koi_disposition']
validation_size = 0.20
seed = 7 
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size = validation_size, random_state =seed)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
scoring = 'accuracy'

models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K-Nearest Neighbors', KNeighborsClassifier()))
models.append(('Classification and Regression Trees', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('Support Vector Machines', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
