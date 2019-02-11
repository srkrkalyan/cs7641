#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:45:52 2019

@author: kalyantulabandu
"""

# Importing libraries
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
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8,random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting classifier to training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=10,max_leaf_nodes=10,min_samples_split=3,splitter='random')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn import metrics
print("Decision Tree model accuracy: ", round(metrics.accuracy_score(y_test,y_pred),6))

from sklearn.model_selection import learning_curve

# Learning Curve - Accuracy

train_sizes = [500,1000,5000,10000,20000]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   DecisionTreeClassifier(max_depth=10,max_leaf_nodes=10,min_samples_split=3,splitter='random'), X,
                                                   y, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'accuracy', 
                                                   shuffle = 'True')


print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

import matplotlib.pyplot as plt
#%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training Accuracy')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation Accuracy')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision Tree (with pruning) - Travel Insurance Data Set', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0.6,1.3)


# Learning Curve - Mean Squared Error

train_sizes =[500,1000,5000,10000,20000]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   DecisionTreeClassifier(max_depth=10,max_leaf_nodes=10,min_samples_split=3,splitter='random'), X,
                                                   y, train_sizes = train_sizes, cv = 5,
                                                   scoring = 'neg_mean_squared_error', 
                                                   shuffle = 'True')


print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)

train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))

import matplotlib.pyplot as plt
#%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training Error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation Error')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Decision Tree (with pruning) - Travel Insurance Data Set', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,0.5)



'''
#Attempting to picturise generated decision tree, so that i can see how decision tree changed after pruning

from sklearn.externals.six import StringIO  
#from IPython.display import Image  
from sklearn.tree import export_graphviz
#import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file='dot_data.dot',  
                filled=True, rounded=True,
                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

'''