# Homework 5 for Cynthia Rudin's Machine Learning class
# All State challenge from Kaggle
# Jeremy Fox

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import csv

# Remove ids from a dataframe and return the array of ids
def strip_ids(df):
    ids = df['id']
    del df['id']
    return ids
# Remove both ids and Y values from a dataframe and return a tuple of the arrays, where
# ids comes first, and Y values come second
def strip_Y_ids(df):
    ids = df['id']
    del df['id']
    Y = df['loss'].tolist()
    del df['loss']
    return (ids, Y)
# return an array containing all the categorical labels in a dataframe
def get_categorical_keys(df):
    return [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['O']]
# return an array containing the indices of all the categorical labels in a dataframe
def get_categorical_indices(df):
    return [i for i, val in enumerate(df.dtypes) if val == "O"]

# Read in training data
def read_train_data(filename):
    # Reads in data, using headers for column names and inferring the type
    df = pd.read_csv(filename)

    # delete id column and save and delete Y values
    ids, Y = strip_Y_ids(df)

    # get all categorical columns
    keys = get_categorical_keys(df)
    indices = get_categorical_indices(df)

    # convert categorical data to numeric using label encoders, store Label Encoders for later use
    les = {}
    for key in keys:
        col = df[key]
        le = preprocessing.LabelEncoder()
        le.fit(col)
        df[key] = le.transform(col)
        les[key] = le


    # convert dataframe to matrix
    matrix = df.as_matrix()

    # encode data using onehot encoder, store the encoder
    enc = preprocessing.OneHotEncoder(categorical_features=indices)
    enc.fit(matrix)
    X = enc.transform(matrix)

    # drop features based on mutual information
    selector = SelectKBest(mutual_info_regression, k=100)
    selector.fit(X,Y)
    selector.transform(X)

    # Here, X is the training data, Y is the labels, les is a set of LabelEncoders,
    # enc is the OneHotEncoder, and selector is the feature selector
    return X, Y, les, enc, selector

# Read in test data. Requires LabelEncoders, OneHotEncoder, and selector used in reading the training data.
def read_test_data(filename, les, enc, selector):
    df = pd.read_csv(filename)
    ids = strip_ids(df)

    keys = get_categorical_keys(df)
    indices = get_categorical_indices(df)

    # convert categorical data to numeric using previous label encoders

    for key in keys:
        col = df[key]
        le = les[key]
        # prevent new values from showing up. This is important because if a value is in a category
        # in the test data and was not in the training data, then that will cause the algorithm to break.
        # Here, I am making an assumption -- there will not be too many new values, and they will not
        # be that statistically significant, so I can make them equal to a different value and it will not
        # mess up the model
        col = map(lambda s: col[0] if s not in le.classes_ else s, col)
        df[key] = le.transform(col)


    matrix = df.as_matrix()
    print matrix.shape
    # Encode using OneHotEncoder
    X = enc.transform(matrix)
    # drop same features as used in training
    X = selector.transform(X)

    # Here, X is the training data, and ids are the data ids. The ids are needed to write
    # the data out as a csv
    return X, ids



# Rotates an np row into an np column.
def row_to_column(row):
    return np.reshape(row,(row.shape[0],1))

# returns a list of tuples of a datapoint and its label
def combine_data_and_labels(data, labels):
    return zip(data,labels)

# writes out a matrix as a csv at given file path. Uses a comma as delimiter
def write_csv(matrix, filepath):
    ofile = open(filepath,"wb")
    writer = csv.writer(ofile,delimiter=',')
    for row in matrix:
        writer.writerow(row)
    ofile.close()

# Begin main code body

X, y, les, enc, selector = read_train_data("pml_train.csv")

# Set up to use different algorithms

# Set up MLPRegressor
mlp_parameters = {'hidden_layer_sizes': [(1,),(10,),(50,),(100,)], 'activation': ['identity', 'logistic', 'tanh']}
mlp_algo = GridSearchCV(MLPRegressor(max_iter=1000), mlp_parameters, cv=5)

# Set up Ridge Regression
ridge_parameters = {'alpha': [1, 10, 100, 1000]}
ridge_algo = GridSearchCV(Ridge(fit_intercept=True), ridge_parameters, cv=5)

# Set up LASSO
lasso_parameters = {'alpha': [1, 10, 100, 1000]}
lasso_algo = GridSearchCV(Lasso(fit_intercept=True), ridge_parameters, cv=5)

algos = [(mlp_algo, "mlp"), (ridge_algo, "ridge"), (lasso_algo, "lasso")]

# Use cross validation to prevent overfitting
for pair in algos:
    algo = pair[0]
    label = pair[1]
    print "Computing cross validation scores..."
    scores = cross_val_score(algo, X, y, cv=5, scoring=mean_absolute_error)
    print "Printing scores for " + label
    for score in scores:
        print score

# Split the dataset into two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# Read test data
test_X, ids = read_test_data("pml_test_features.csv", les, enc, selector)

# convert ids to integers
ids = ids.astype(int)

# Do actual training with the different algorithms compute metrics, and write out a csv
for pair in algos:
    algo = pair[0]
    label = pair[1]
    # fit and predict
    algo.fit(X_train,y_train)
    y_true, y_pred = y_test, reg.predict(X_test)

    # Print out metrics
    print explained_variance_score(y_true, y_pred)
    print mean_absolute_error(y_true, y_pred)
    print r2_score(y_true, y_pred)

    # write out csv file
    out_matrix = combine_data_and_labels(ids, y_pred)
    write_csv(out_matrix,label + " out.csv")

print "finished"
