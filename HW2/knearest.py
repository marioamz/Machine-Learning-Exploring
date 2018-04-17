#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:50:04 2018

@author: mariomoreno
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def k_neighbors_model(df, dependent_var, size_test):
    '''
    This runs the machine learning K-Nearest Neighbors algorithm. 
    
    It loops through the 10 nearest neighbors, four different methods,
    and two different weights to find and return the most accurate model.
    '''
    
    x_train, x_test, y_train, y_test = split_data(df, dependent_var, size_test)
    
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    weights = ['uniform', 'distance']
    models = []

    for k in range(10):
        for metric in metrics:
            for weight in weights:
                knn_loop = KNeighborsClassifier(n_neighbors=(k + 1), weights=weight, metric=metric)
                knn_loop.fit(x_train, y_train)
                results = knn_loop.predict(x_test)
                
                accuracy = accuracy_score(y_test, results, normalize = True, sample_weight = None)
                models.append({'Metric': metric, 'Weight': weight, 'Neighbors': (k+1), 'Accuracy': accuracy})
                
    models_df = pd.DataFrame(models)
    best_index = evaluate(models_df)
    
    return models_df.loc[best_index]


def split_data(df, dep_variable, test):
    '''
    This function splits the data into train/test pairs
    '''
    
    dependent = pd.DataFrame(df[dep_variable])
    del df[dep_variable]
    
    return train_test_split(df, dependent, test_size=test)


def evaluate(results):
    '''
    This function finds the most accurate of all the models we tested.
    '''
    
    return results['Accuracy'].idxmax()
