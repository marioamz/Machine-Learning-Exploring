#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:50:04 2018

@author: mariomoreno
"""

from __future__ import division
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



import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm, model_selection
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
np.random.seed(524)


def define_clfs_grids(grid_size):

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    large_grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    test_grid = {
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

    if grid_size == 'large':
        return clfs, large_grid
    elif grid_size == 'small':
        return clfs, small_grid
    elif grid_size == 'test':
        return clfs, test_grid


def clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test):


    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                #print(p)
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(x_train, y_train.ravel()).predict_proba(x_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    print(y_pred_probs_sorted, y_test_sorted)
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    #continue
    return results_df



def plot_precision_recall_n(y_true, y_prob, model_name, filename_short):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    plt.rcParams.update({'font.size': '9'})
    plt.rcParams.update({'figure.dpi': '300'})
    plt.rcParams.update({'figure.figsize': '16, 12'})
    plt.title(model_name)
    filename = "results/PR_curve_{}.png".format(filename_short)
    plt.savefig(filename)
    plt.close('all')


def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)


def go(x_train, x_test, y_train, y_test, grid_size, models_to_run):

    # define grid to use: test, small, large
    clfs, grid = define_clfs_grids(grid_size)

    # define models to run
    #models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', ]

    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test)

    return results_df
