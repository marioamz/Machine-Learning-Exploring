"""
Created on Mon Apr  9 17:47:05 2018

@author: mariomoreno
"""

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


def read_data(csv_file, rnm=0):
    '''
    This function reads in the data, then calls functions to explore it,
    clean it, and then returns a cleaned dataframe.
        - csv_file = file to be read in
        - rnm = 0 if columns don't need to be renamed, 1 if they do
    '''

    unclean = pd.read_csv(csv_file)

    if rnm == 1:
        unclean.columns = [rename_columns(col) for col in unclean]
    else:
        return unclean

    return unclean


def rename_columns(column_name):
    '''
    This function takes in a column to be renamed and turns it
    into a more readable snake form.
    '''

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def explore_data(df):
    '''
    This function runs a series of basic exploratory commands, including:
        - Tail and Head
        - Summary Stats
        - Null Values
        - Histograms

    It then calls functions that impute null values, remove instances of
    multicollinearity, and discretize or turn varibales into binary numbers
    '''

    # Summary stats for each column
    for i in df:
        print()
        print('Summary stats for', i)
        print(df[i].describe())
        print()

    # Find nulls
    nulls(df)

    #Histograms for every column
    plot_data(df, 'hist', None, None)


def split_data(df, dep_variable, test_size, by_time, date, date_ranges):
    '''
    This function splits the data into train/test pairs

    If we're working on a time bound project, it does so by
    period of time based on the date ranges dictionary. Otherwise
    it just uses the sklearn split function

        - df = dataframe upon which to build the test-train split
        - dep_variable = the y variable
        - test_size = if by_time is False, this is the test size split
        - by_time = if we're working in a time variant project
        - date_ranges = empty if not a time variant project, otherwise it's the
                        train test split for cross validation
        - date = date column that we need to track if in time variant project

    In time variant, the last tuple in the list is the train test pair that
    you want to do your final analysis on. The others are the splits upon the
    training set.
    '''

    if by_time == False:

        dependent = pd.DataFrame(df[dep_variable])
        del df[dep_variable]

        x_train, x_test, y_train, y_test = train_test_split(df, dependent, test_size=test)

        return x_train, x_test, y_train, y_test

    elif by_time:
        time_splits = []

        for_test_df = df.loc[:, [dep_variable, date]]
        del df[dep_variable]

        for k, v in date_ranges.items():
            #datetime.strptime(k[0], )
            x_train = (df[date] > k[0]) & (df[date] <= k[1])
            x_test = (for_test_df[date] > k[0]) & (for_test_df[date] <= k[1])
            y_train = (df[date] > v[0]) & (df[date] <= v[1])
            y_test = (for_test_df[date] > v[0]) & (for_test_df[date] <= v[1])

            x_train_df = df.loc[x_train]
            x_test_df = for_test_df.loc[x_test]
            y_train_df = df.loc[y_train]
            y_test_df = for_test_df.loc[y_test]

            del y_test_df[date]
            del x_test_df[date]
            del x_train_df[date]
            del y_train_df[date]

            time_splits.append((x_train_df, x_test_df, y_train_df, y_test_df))

        return time_splits


def analyze_split_data(df_x_train, df_y_train):
    '''
    This function dives deeper into our train, test splits.

    In doing so, we hope to determine which variables need to be
    turned into binary variables, which ones need to be discretized,
    which ones to delete based on correlations, and more.
    '''

    # Call same explore data function as before on train and test
    print('------ Analysis for X Training Set ------')
    explore_data(df_x_train)

    print('------ Analysis for Y Training Set ------')
    explore_data(df_y_train)


def explore_potential_correlations(df):
    '''
    This function explores potential correlations between variables

    This is to find instances of multicollinearity.
    '''

    axis = plt.axes()
    sns.heatmap(df.corr(), square=True, cmap='PiYG')
    axis.set_title('Correlation Matrix')
    plt.show()


def dummify(df, to_dummy):
    '''
    This function takes a list of columns to turn from categorical
    to dummy variable features
    '''

    new_df = pd.get_dummies(df, columns = to_dummy, dummy_na=True)

    return new_df


def nulls(df):
    '''
    This functions finds columns that might have nulls
    '''

    print('Values with True have nulls')
    print(df.isnull().any())
    print()


def clean_data(df, nulls, discretize, binary, to_del):
    '''
    This function cleans the data by removing variables that have
    multicollinearity, filling any nulls, and discretizing/turning
    variables to binary.
        - df = dataframe
        - nulls = dictionary where keys are variables and values is how
                    to impute (zero, mean, median)
        - discretize = a list of variables to discretize
        - binary = a list of variables to turn into binary
        - to_del = a list of variables to be dropped

    '''

    #Nulls
    for key, val in nulls.items():
        if val == 'zero':
            df[key].fillna(0, inplace=True)
        elif val == 'mean':
            df[key].fillna(df[key].mean(), inplace=True)
        elif val == 'median':
            df[key].fillna(df[key].median(), inplace=True)

    #Discretize
    for var in discretize:
        to_discretize(df, var)

    #Binary
    for binary_var in binary:
        to_binary(df, binary_var)

    #Delete
    for delvar in to_del:
        del df[delvar]

    return df

def to_discretize(df, var):
    '''
    This function discretizes variables into more workable ranges.

    The ranges are not automated.
    '''


    student_bins = range(0, 1000, 100)
    price_bins = range(0, 3500, 100)

    if var == 'students_reached':
        df['students_reached_groups'] = pd.cut(df[var], student_bins, labels=False)
        del df[var]
    elif var == 'total_price_excluding_optional_support':
        df['tp_exclude'] = pd.cut(df[var], price_bins, labels=False)
        del df[var]
    elif var == 'total_price_including_optional_support':
        df['tp_include'] = pd.cut(df[var], price_bins, labels=False)
        del df[var]


def remove_outliers(df, cols):
    '''
    This function removes anything outside of three standard deviations within
    a column
    '''

    df = df[((df[cols] - df[cols].mean()) / df[cols].std()).abs() < 3]
    return df


def to_binary(df, var):
    '''
    This function turns a column of continous variables into binary
    for ease of processing
    '''

    new_col = str(var) + '_binary'
    df[new_col] = (df[var] >= df[var].mean()).astype(int)

    del df[var]


def true_to_false(df, cols):
    '''
    This function takes the dataframe and a list of columns with just
    True or False values, and turns True to 1 and False to 0
    '''

    label_enc = preprocessing.LabelEncoder()

    for c in cols:
        label_enc.fit(df[c])
        new_var = label_enc.transform(df[c])
        new_name = c + '_new'
        df[new_name] = new_var
        del df[c]


def plot_data(df, plot_type, var1, var2):
    '''
    This function builds a few simple plots to visualize and start to
    understand the data. Can be called on the cleaned data, or on the
    original dataframe.
    '''

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

    elif plot_type == 'pie':
        print('Pie Chart for', var1)
        df.plot.pie(y = var1, figsize=(20,10))
        plt.show()
# Things to add over time: distribution graphs
