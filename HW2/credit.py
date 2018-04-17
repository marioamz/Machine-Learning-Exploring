#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:47:05 2018

@author: mariomoreno
"""

import pandas as pd
import matplotlib.pyplot as plt

def read_data(csv_file):
    '''
    This function reads in the data, then calls functions to explore it,
    clean it, and then returns a cleaned dataframe.
    '''

    credit = pd.read_csv(csv_file)

    credit.rename(columns={"SeriousDlqin2yrs": "experienced_90days_delinquency", \
                    "NumberOfTime30-59DaysPastDueNotWorse": "times_30-59_pastdue", \
                    "DebtRatio": "debt_ratio", \
                    "MonthlyIncome": "monthly_income", \
                    "NumberOfOpenCreditLinesAndLoans": "times_open_credit_loans",\
                    "NumberOfTimes90DaysLate": "times_late",
                    "NumberRealEstateLoansOrLines": "realestate_loans_lines", \
                    "NumberOfTime60-89DaysPastDueNotWorse": "times_60-89_pastdue",
                    "NumberOfDependents": "dependents",\
                    "RevolvingUtilizationOfUnsecuredLines": "use_unsecured_lines"
                    }, inplace = True)


    return credit


def explore_data(df):
    '''
    This function runs a series of basic exploratory commands, including:
        - Tail and Head
        - Summary Stats
        - Value Counts
        - Correlations
        - Null Values

    It then calls functions that impute null values, remove instances of
    multicollinearity, and discretize or turn varibales into binary numbers
    '''

    # Head and Tail of entire dataframe
    print()
    print('Top 5 and bottom 5 rows')
    print(df.head())
    print(df.tail())
    print()

    # Summary stats for each column
    for i in df:
        print()
        print(i, 'Summary stats for', i)
        print(df[i].describe())
        print()

    # Value counts for each column
    for j in df:
        print()
        print('Values that make up', j)
        print(df[j].value_counts())
        print()

    # Correlations between variables
    explore_potential_correlations(df)

    # Find nulls
    nulls(df)


def explore_potential_correlations(df):
    '''
    This function explores potential correlations between variables

    This is to find instances of multicollinearity.
    '''

    for feature in df:
        print()
        print('Correlations for:', feature)
        print()
        print(df.corr().unstack()[feature])
        print()


def nulls(df):
    '''
    This functions finds columns that might have nulls
    '''

    print('Values with True have nulls')
    print(df.isnull().any())
    print()


def clean_data(df):
    '''
    This function cleans the data by removing variables that have
    multicollinearity, filling any nulls, and discretizing/turning
    variables to binary.

    This process is not automated, since filling in nulls, multicollinearity,
    and other processes will depend on the data itself.
    '''

    #From nulls, found that dependents and monthly_income have nulls
    df['dependents'].fillna(0, inplace=True)
    df['monthly_income'].fillna(df['monthly_income'].median(), inplace=True)

    #From correlations function, learned that there is multicollinearity
    del df['times_30-59_pastdue']
    del df['times_60-89_pastdue']

    #From plots, common-sense, decided to discretize age and monthly_income
    to_discretize_one = 'age'
    to_discretize_two = 'monthly_income'

    discretize(df, to_discretize_one)
    discretize(df, to_discretize_two)

    #From data exploration, decided to turn times_late to binary
    to_binary = 'times_late'
    binary(df, to_binary)
    
    # Delete one last useless variable
    del df['PersonID']
    
    df['monthly_income_groups'].fillna(0, inplace=True)
    df['age_groups'].fillna(0, inplace=True)

    return df

def discretize(df, var):
    '''
    This function discretizes variables into more workable ranges.

    The ranges are not automated.
    '''

    age_bins = range(0, 110, 10)
    income_bins = range(0, 200000, 10000)

    if var == 'age':
        df['age_groups'] = pd.cut(df[var], age_bins, labels=False)
    elif var == 'monthly_income':
        df['monthly_income_groups'] = pd.cut(df[var], income_bins, labels=False)

    del df[var]


def binary(df, var):
    '''
    This function turns a column of continous variables into binary
    for ease of processing
    '''

    df['times_late_binary'] = (df[var] >= df[var].mean()).astype(int)

    del df[var]

def plot_data(df, plot_type, var1, var2):
    '''
    This function builds a few simple plots to visualize and start to
    understand the data. Can be called on the cleaned data, or on the
    original dataframe.
    '''

    #Called on cleaned data in this instance.

    if plot_type == 'hist':
        print('Histograms of all columns')
        df.hist(figsize=(20,20))
        plt.show()

    elif plot_type == 'bar':
        print('Bar Chart for', var1)
        df[var1].value_counts().plot('bar', figsize=(20,10))
        plt.show()

    elif plot_type == 'scatter':
        print('Scatter plot between', var1, 'and', var2)
        plt.scatter(df[var1], df[var2])
        plt.show()

    elif plot_type == 'line':
        print('Line graph between', var1, 'and', var2)
        df[[var1, var2]].groupby(var1).mean().plot(figsize=(20,10))
        plt.show()
