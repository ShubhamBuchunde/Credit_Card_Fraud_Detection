# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:34:11 2019

@author: Shubham Buchunde
"""


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


print('Python_version--->{}'.format(sys.version))

data = pd.read_csv('creditcard.csv')
data.head()
data.shape
data.columns
data.describe
data.info
data.isnull().values.any()

#Minimizing the data
data = data.sample(frac = 0.1,random_state=1)
data.shape

Count_V_F = pd.value_counts(data['Class'], sort = True)
print(Count_V_F)
Count_V_F.plot(kind = 'bar', rot=0)

plt.title("Class Vs Frequency Bar graph")
plt.xlabel("Class")
plt.ylabel("Frequency")

data.hist(figsize=(20,20))
plt.show()

# correlation martrix
corrmat= data.corr()
plt.figure(figsize =(9,9))
fig =sns.heatmap(corrmat,cmap="BuPu",vmax= 0.8,square = True)
plt.show()

Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]
outlier_function = len(Fraud)/float(len(Valid))
print(outlier_function)
print(len(Fraud))
print(len(Valid))

x = data.iloc[:,:-1].values
y= data.iloc[:,-1].values

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

classifiers = {
        "Isolation Forest": IsolationForest(max_samples= len(x),
                                            contamination = outlier_function,
                                            random_state = 1),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                                   contamination = outlier_function)
        
        }

for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name ==  "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        score_pred= clf.negative_outlier_factor_
    else:
        clf.fit(x)
        score_pred= clf.decision_function(x)
        y_pred = clf.predict(x)
        
    #Assigning the correct values of fraud and valid
    y_pred[y_pred== 1] = 0
    y_pred[y_pred== -1] = 1
    
    n_errrors = (y_pred!=y).sum()
    
    print('{}..{}'.format(clf_name,clf))
    print(accuracy_score(y_pred,y))
    print(classification_report(y_pred,y))
    
    
    

















