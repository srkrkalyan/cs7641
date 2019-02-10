#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:17:06 2019

@author: kalyantulabandu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('travel_insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder()
X[:,0] = labelencoder_X_0.fit_transform(X[:,0])

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

labelencoder_X_3 = LabelEncoder()
X[:,3] = labelencoder_X_3.fit_transform(X[:,3])

labelencoder_X_5 = LabelEncoder()
X[:,5] = labelencoder_X_5.fit_transform(X[:,5])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


onehotencoder_0 = OneHotEncoder(categorical_features = [0,1,2,3,5])
X = onehotencoder_0.fit_transform(X).toarray()

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 1)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn import metrics
print("KNN model accuracy: ", round(metrics.accuracy_score(y_test,y_pred),4))
