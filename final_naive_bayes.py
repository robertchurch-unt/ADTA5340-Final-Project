#!/usr/bin/python3.7

'''
    File: sample_dataset.py
    Author: Robert Church robert@robertchurch.us
    Date: 2022 July 24
    Course: ADTA 5340
    Assignment: Final Project
    
    Purpose:  Purpose of this script is to take a sample file
              and perform naive bayes analysis against three columns.
              
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


#
# Establish dataframe from input file
#

df = pd.read_csv('flow_sample.csv')

#
# Establish array of only the columns we need
#

df2 = df[['duration','packet count','byte count']]

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
# Due to Memory limitations, this had to be set to 20% / 80%
#

test_size = 0.2

#
# Randomly select X records within each sub-dataset
#

seed = 7

#
# Split the dataset (input and output) into training / test datasets
#

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size, random_state = seed)

#
# Feature Scaling
#
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#
# Training the Naive Bayes model on the Training set
#

classifier = GaussianNB()
classifier.fit(X_train, y_train)

#
# Compute training and test data accuracy result
#
y_pred = classifier.predict(X_test)

#
# Produce confusion matrix
#
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

#
# Output accuracy score
#

print("\n\n")
print("Accuracy Score: ", ac)
print("\n\n")
