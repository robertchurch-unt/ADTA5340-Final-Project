#!/usr/bin/python3.7

'''
    File: final_knearest.py
    Author: Robert Church robert@robertchurch.us
    Date: 2022 July 07
    Course: ADTA 5340
    Assignment: Final Project
    
    Purpose:  Purpose of this script is to take a sample file
              and perform k-nearest analysis against three columns.
              
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size, random_state=seed)

#
# Build the model
#

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

#
# Compute training and test data accuracy for each value of k
#

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

#
# Generate plot
#

plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.savefig('k-nearest.jpg')
