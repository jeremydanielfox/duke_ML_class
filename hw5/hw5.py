# Homework 5 for Cynthia Rudin's Machine Learning class
# All State challenge from Kaggle
# Jeremy Fox

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import csv

def strip_ids(df):
    ids = df['id']
    del df['id']
    return ids

def strip_Y_ids(df):
    ids = df['id']
    del df['id']
    Y = df['loss'].tolist()
    del df['loss']
    return (ids, Y)

def get_categorical_keys(df):
    return [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['O']]

def get_categorical_indices(df):
    return [i for i, val in enumerate(df.dtypes) if val == "O"]

# Read in data
def read_test_data(filename, les, enc):
    df = pd.read_csv(filename)
    ids = strip_ids(df)

    keys = get_categorical_keys(df)
    indices = get_categorical_indices(df)

        # convert categorical data to numeric using previous label encoders

    for key in keys:
        col = df[key]
        le = les[key]
        # prevent new values from showing up
        col = map(lambda s: col[0] if s not in le.classes_ else s, col)
        df[key] = le.transform(col)


    matrix = df.as_matrix()
    print matrix.shape
    # Encode using OneHotEncoder
    X = enc.transform(matrix)
    return X, ids


def read_train_data(filename):
    # Reads in data, using headers for column names and inferring the type
    df = pd.read_csv(filename)

    # delete id column and save and delete Y values
    ids, Y = strip_Y_ids(df)

    # get all categorical columns
    keys = get_categorical_keys(df)
    indices = get_categorical_indices(df)
        # convert categorical data to numeric, store Label Encoders for later use
    les = {}
    for key in keys:
        col = df[key]
        le = preprocessing.LabelEncoder()
        le.fit(col)
        df[key] = le.transform(col)
        les[key] = le



    matrix = df.as_matrix()
    print matrix.shape
    # encode data using onehot encoder, store the encoder
    enc = preprocessing.OneHotEncoder(categorical_features=indices)
    enc.fit(matrix)
    X = enc.transform(matrix)


    # save labels and remove them from data
   # if not test:
    #    Y = out[:,-1]
     #   out = np.delete(out,-1,1)

    # standardize data
  #  out = preprocessing.scale(out)
   # if test:
    #    return out, ids
    return X, Y, les, enc

def row_to_column(row):
    return np.reshape(row,(row.shape[0],1))

def combine_data_and_labels(data, labels):
    #data = row_to_column(data)
    #labels = row_to_column(labels)
    #out = np.append(data,labels,axis=1)
    return zip(data,labels)

def print_results(ids, Y):
    ids = ids.astype(int)
    out_matrix = combine_data_and_labels(ids, Y)
    write_csv(out_matrix,"out.csv")

def write_csv(matrix, filepath):
    ofile = open(filepath,"wb")
    writer = csv.writer(ofile,delimiter=',')
    for row in matrix:
        writer.writerow(row)
    ofile.close()


X, y, les, enc= read_train_data("pml_train.csv")

# Split the dataset into two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# clf = GridSearchCV(Ridge(fit_intercept=True), tuned_parameters, cv=5)
reg = MLPRegressor(hidden_layer_sizes=(15,), max_iter=200)
print "shape training"
print X_train.shape
reg.fit(X_train,y_train)
print "Neural net..."
print "report"
print "shape"

y_true, y_pred = y_test, reg.predict(X_test)

residuals = [a_i - b_i for a_i, b_i in zip(y_true, y_pred)]

out_matrix = combine_data_and_labels(y_pred, residuals)

print "printing residuals...    "
write_csv(out_matrix,"residuals.csv")


print explained_variance_score(y_true, y_pred)
print mean_absolute_error(y_true, y_pred)
print r2_score(y_true, y_pred)

# Write out csv to turn in

# Y_final = reg.predict(X)

print "Kernel Ridge Regression Regression..."
#clf = KernelRidge(alpha=100, kernel='rbf')
#clf = GridSearchCV(KernelRidge(kernel='rbf'), tuned_parameters, cv=5)
#clf = LinearRegression(fit_intercept=True)
#clf.fit(X_train, y_train)
print "Best parameters found on development set:"
#print clf.best_params_
print ""
print "Detailed classification report:"
print ""
print "The model is trained on the full development set."
print "The scores are computed on the full evaluation set."
print ""
#y_true, y_pred = y_test, clf.predict(X_test)
#print explained_variance_score(y_true, y_pred)
#print mean_absolute_error(y_true, y_pred)
#print r2_score(y_true, y_pred)

#reg = MLPRegressor(hidden_layer_sizes=(15,))
#reg.fit(data,y)
#print "fitted"

test_X, ids = read_test_data("pml_test_features.csv", les, enc)

# convert ids to integers
ids = ids.astype(int)
print "shape"
print test_X.shape
test_labels = reg.predict(test_X)
out_matrix = combine_data_and_labels(ids, test_labels)

write_csv(out_matrix,"out.csv")

print "finished"
