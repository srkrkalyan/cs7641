#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:34:56 2019

@author: kalyantulabandu
"""

# Importing libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast_cancer_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding categorical data
# No categorical data, so no encoding needed

# splitting X & y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fitting classifier to training set
from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy',splitter='best')
classifier = DecisionTreeClassifier(max_depth=10,max_leaf_nodes=10,min_samples_split=3,splitter='random')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn import metrics
print("Decision Tree model accuracy: ", round(metrics.accuracy_score(y_test,y_pred),6))

#values of hyperparameters
max_depth = [3,5,8,10]
min_samples_split = [2,3,4,5]
splitter = ['best','random']
criteria = ['entropy','gini']
max_leaf_nodes = [8,10,15,20]
hyperparameters = dict(max_depth=max_depth,min_samples_split=min_samples_split,splitter=splitter,max_leaf_nodes=max_leaf_nodes)


# training knn model on training set
from sklearn.model_selection import GridSearchCV
import time

# Hyperparameter generation using grid search
decision_classifier = DecisionTreeClassifier()
grid = GridSearchCV(estimator=decision_classifier, param_grid=hyperparameters, cv=5, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(X_train,y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

'''
#Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random = RandomizedSearchCV(estimator=knn_classifier, param_distributions=hyperparameters, cv=3, n_jobs=-1)
start_time = time.time()
random_result = random.fit(X, y)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

#Attempting to picturise generated decision tree, so that i can see how decision tree changed after pruning

from sklearn.externals.six import StringIO  
#from IPython.display import Image  
from sklearn.tree import export_graphviz
#import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file='dt_data_breast_cancer.dot',  
                filled=True, rounded=True,
                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())
'''


# Learning Curve - Accuracy

train_sizes = [10,30,60,70,100,150,180,200,220,260,300,350]

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
plt.title('Learning curves for Decision Tree (with pruning) - Breast Cancer Data Set', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,3)


# Learning Curve - Mean Squared Error

train_sizes = [10,30,60,70,100,150,180,200,220,260,300,350]

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
plt.title('Learning curves for Decision Tree (with pruning) - Breast Cancer Data Set', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,3)