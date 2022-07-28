#!/usr/bin/python3.7

'''
    File: final_regression.py
    Author: Robert Church robert@robertchurch.us
    Date: 2022 July 24
    Course: ADTA 5340
    Assignment: Final Project
    
    Purpose:  Purpose of this script is to take a sample file
              and perform regression analysis against three
              columns.
              
              Note: There is no correlation between duration and packet size,
              but the task wasn't to prove anything, just to perform the
              regression.

    Input:    The file flows_sample.csv was obtained from the file flows.tar.gz
              obtained here: https://csr.lanl.gov/data/cyber1/
    Output:   The script will output a classification report, and the count of
              accurate hits.
              
'''

#
# Import necessary libraries
#

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
import warnings
warnings.filterwarnings('ignore')

#
# Establish dataframe from input file
#

input_file = 'flow_sample.csv'
df = pd.read_csv(input_file)

#
# Display head of dataframe to show the file loaded
#

df.head()

#
# Establish array of only the columns we need
#

df2 = df[['duration','packet count','byte count']]

#
# Set some optimization for sklearn
#

sklearn.set_config(assume_finite=True,working_memory=4096)

#
# store dataframe values into a numpy array
#

array = df2.values

#
# Split array between independent and dependent columns (input and output)
#
# X = independent variables
#     duration, packet count
# Y = dependent variables
#     byte count
#

X = array[:,0:1]
Y = array[:,2]

#
# Establish the number of records to use as the training subset vs testing subset
#
# Due to Memory limitations, this had to be set to 50% / 50%
#

test_size = 0.5

#
# Randomly select X records within each sub-dataset
#

seed = 7

#
# Split the dataset (input and output) into training / test datasets
#

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=test_size,random_state=seed)

#
# Build the model
#

model = LogisticRegression(random_state=seed, max_iter=50)

#
# Train the model using the training sub-dataset
#

model.fit(X_train, Y_train)

#
# Use the magic wand enable multithreading to speed this model up (threading to 6) (8 cpu 32G memory)
#

with parallel_backend('threading', n_jobs=6):
    #print the classification report
    predicted = model.predict(X_test)
    report = classification_report(Y_test, predicted)

#
# Output the classification report
#

print("Classification Report: ", "\n", "\n",report)

#
# Output overall success rate
#

print("Total measured: ", Y_test)
print("Total accurate: ", accuracy_score(Y_test, predicted, normalize=False))
