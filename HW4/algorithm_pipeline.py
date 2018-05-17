#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:50:04 2018

@author: mariomoreno
"""


import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def k_neighbors_model(x_train, x_test, y_train, y_test):
    '''
    This runs the machine learning K-Nearest Neighbors algorithm.

    It loops through the 10 nearest neighbors, four different methods,
    and two different weights to find and return the most accurate model.
    '''

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


def decision_tree_model(x_train, x_test, y_train, y_test):
    '''
    This runs through different parameters for decision trees and
    returns the best model
    '''

    depths = [1, 3, 5, 7, 9]
    criterion = ['gini', 'entropy']
    features = [1, 5, 9]
    models_dt = []

    for d in depths:
        for c in criterion:
            for f in features:
                dec_tree = DecisionTreeClassifier(criterion=c, max_features=f, max_depth=d)
                dec_tree.fit(x_train, y_train)
                train_pred = dec_tree.predict(x_train)
                test_pred = dec_tree.predict(x_test)
    # evaluate accuracy
                train_acc = accuracy(train_pred, y_train)
                test_acc = accuracy(test_pred, y_test)
                models_dt.append({'Depth': d, 'Features': f, 'Criterion': c, 'Test Accuracy': test_acc})
            
    models_df_dt = pd.DataFrame(models_dt)
    best_index = evaluate(models_df_dt)

    return models_df_dt.loc[best_index]


def split_data(df, dep_variable, test):
    '''
    This function splits the data into train/test pairs
    '''

    dependent = pd.DataFrame(df[dep_variable])
    del df[dep_variable]

    x_train, x_test, y_train, y_test = train_test_split(df, dependent, test_size=test)

    return x_train, x_test, y_train, y_test

def evaluate(results):
    '''
    This function finds the most accurate of all the models we tested.
    '''

    return results['Accuracy'].idxmax()
