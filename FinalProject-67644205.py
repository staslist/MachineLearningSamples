# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:15:15 2016

@author: Stanislav Listopad
CS 284A
Final Project
"""

import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#print('The scikit-learn version is {}.'.format(sklearn.__version__))

import time
# Helper Functions

# Input y: a 1D matrix
# Output z: a scalar value
def one_hot_to_decimal_secondary_structure(y):
    if(y[0] == 1):
        z = 22
    elif(y[1] == 1):
        z = 23
    elif(y[2] == 1):
        z = 24
    elif(y[3] == 1):
        z = 25
    elif(y[4] == 1):
        z = 26
    elif(y[5] == 1):
        z = 27
    elif(y[6] == 1):
        z = 28
    elif(y[7] == 1):
        z = 29
    elif(y[8] == 1):
        z = 30
            
    return z
    
# Input x: a 1D matrix
# Output z: a scalar value
def one_hot_to_decimal_amino_acid_sequence(x):

    if(x[0] == 1):
        z = 0
    elif(x[1] == 1):
        z = 1
    elif(x[2] == 1):
        z = 2
    elif(x[3] == 1):
        z = 3
    elif(x[4] == 1):
        z = 4
    elif(x[5] == 1):
        z = 5
    elif(x[6] == 1):
        z = 6
    elif(x[7] == 1):
        z = 7
    elif(x[8] == 1):
        z = 8
    elif(x[9] == 1):
        z = 9
    elif(x[10] == 1):
        z = 10
    elif(x[11] == 1):
        z = 11
    elif(x[12] == 1):
        z = 12
    elif(x[13] == 1):
        z = 13
    elif(x[14] == 1):
        z = 14
    elif(x[15] == 1):
        z = 15
    elif(x[16] == 1):
        z = 16
    elif(x[17] == 1):
        z = 17
    elif(x[18] == 1):
        z = 18
    elif(x[19] == 1):
        z = 19
    elif(x[20] == 1):
        z = 20
    elif(x[21] == 1):
        z = 21
            
    return z
    
#remove nonsequence values from test dataset    
def endOfSequenceIndex(x, nonseq):
    t = x[(x != nonseq)]
    cutoff = t.shape[0]
    
    return cutoff
    
def classify_KSEQ(dataset_decimal, K, N, M, additional_features_flag, method, method_featureValue):
    
    # N is number of training data samples
    # M is number of testing data samples
    # K is number of amino acids classified as a single secondary structure label
    
    start = time.time()
    
    if(additional_features_flag):
        # Remove the secondary structure label from the input data.
        X_train = np.delete(dataset_decimal[0:N, :, :], 1, 2)
        X_test = np.delete(dataset_decimal[5605:5877, :, :], 1, 2)
    else: 
        # Remove all features except the amino acid label.
        X_train = dataset_decimal[0:N, :, 0:1]
        #print X_train[0, :, 0]
        X_test = dataset_decimal[5605:5877, :, 0:1]

    #L is the number of features
    L = X_train.shape[2]
    y_train = dataset_decimal[0:N, :, 1]
    y_test = dataset_decimal[5605:5877, :, 1]
    
    # USE A SEQUENCE OF K AMINO ACIDS FOR EACH TARGET SECONDARY STRUCTURE LABEL
    
    X_temp_train = np.empty([N, 700, K*L])
    X_temp_test = np.empty([M, 700, K*L])
    
    index_0 = 0
    index_1 = 0
    # For each protein's amino acid (in training data) consolidate features of that and of 2 neighoring amino acids into the features of that amino acid
    while index_0 < N:
        while index_1 < 700:
            # beginning of sequence
            if(index_1 < ((K-1)/2)):
                index_2 = 0
                index_3 = 0
                index_4 = 0
                while index_3 < K*L:
                    X_temp_train[index_0][index_1][index_3] = X_train[index_0, (index_1 + index_2), index_4]
                    index_3 += 1
                    index_4 += 1
                    if(index_3 % L == 0):
                        index_2 += 1
                        index_4 = 0
            # end of sequence
            elif(index_1 >= (700 - ((K-1)/2))):
                index_2 = 0
                index_3 = 0
                index_4 = 0
                while index_3 < K*L:
                    X_temp_train[index_0][index_1][index_3] = X_train[index_0, (index_1 - index_2), index_4]
                    index_3 += 1
                    index_4 += 1
                    if(index_3 % L == 0):
                        index_2 += 1
                        index_4 = 0
            else:
                index_3 = 0
                index_4 = 0
                
                while index_3 < L:
                    X_temp_train[index_0][index_1][index_3] = X_train[index_0, index_1, index_3]
                    index_3 += 1
                    
                z = 1
                index_5 = 1
                
                while index_3 < K*L:
                    X_temp_train[index_0][index_1][index_3] = X_train[index_0, (index_1 + z), index_4]
                    index_3 += 1
                    index_4 += 1
                    
                    if(index_3 % L == 0):
                        index_4 = 0
                        z = -1 * z
                        if(index_5 % 2):
                            z += 1
                        index_5 += 1
            index_1 += 1
        index_1 = 0
        index_0 += 1
    X_train = X_temp_train
    
    index_0 = 0
    index_1 = 0
    # For each protein's amino acid (in testing data) consolidate features of that and of 2 neighoring amino acids into the features of that amino acid
    while index_0 < M:
        while index_1 < 700:
            # beginning of sequence
            if(index_1 < ((K-1)/2)):
                index_2 = 0
                index_3 = 0
                index_4 = 0
                while index_3 < K*L:
                    X_temp_test[index_0][index_1][index_3] = X_test[index_0, (index_1 + index_2), index_4]
                    index_3 += 1
                    index_4 += 1
                    if(index_3 % L == 0):
                        index_2 += 1
                        index_4 = 0
            # end of sequence
            elif(index_1 >= (700 - ((K-1)/2))):
                index_2 = 0
                index_3 = 0
                index_4 = 0
                while index_3 < K*L:
                    X_temp_test[index_0][index_1][index_3] = X_test[index_0, (index_1 - index_2), index_4]
                    index_3 += 1
                    index_4 += 1
                    if(index_3 % L == 0):
                        index_2 += 1
                        index_4 = 0
            else:
                index_3 = 0
                index_4 = 0
                
                while index_3 < L:
                    X_temp_test[index_0][index_1][index_3] = X_test[index_0, index_1, index_3]
                    index_3 += 1
                    
                z = 1
                index_5 = 1
                
                while index_3 < K*L:
                    X_temp_test[index_0][index_1][index_3] = X_test[index_0, (index_1 + z), index_4]
                    index_3 += 1
                    index_4 += 1
                    
                    if(index_3 % L == 0):
                        index_4 = 0
                        z = -1 * z
                        if(index_5 % 2):
                            z += 1
                        index_5 += 1
            index_1 += 1
        index_1 = 0
        index_0 += 1
    X_test = X_temp_test
        
    # Change the X_train into a 2D matrix & the y_train in 1D matrix
    X_train = np.reshape(X_train, ((N * 700), K*L))
    #print X_train.shape
    #print X_train[2, :]
    y_train = np.reshape(y_train, (N * 700))
    
    # REMOVE NON SEQUENCE VALUES FROM TESTING DATA & RESHAPE the X_test matrix in 2D & the y_test matrix in 1D
    cutoff = endOfSequenceIndex(X_test[0, :, 0], 21)
    X_temp = X_test[0, 0:cutoff, :]
    y_temp = y_test[0, 0:cutoff]
    index_0 = 1
    while index_0 < M:
        cutoff = endOfSequenceIndex(y_test[index_0, :], 30)
        X_temp = np.concatenate((X_temp, X_test[index_0, 0:cutoff, :]), axis = 0)
        y_temp = np.concatenate((y_temp, y_test[index_0, 0:cutoff]), axis = 0)
        index_0 += 1
    
    X_test = X_temp
    y_test = y_temp
    
    string_0 = 'APPROACH(3) USE AMINO ACID SEQUENCES OF LENGTH (K = '
    string_1 = ") & ALL_FEATURES_ENABLED = "
    string_3 = "; Machine Learning Method: "
    string_2 = string_0 + str(K) + string_1 + str(additional_features_flag) + string_3 + method
    print(string_2)
    
    if(method == 'LR'):
        logistic = linear_model.LogisticRegression(C = method_featureValue)
        print('LogisticRegression score: %f'
              % logistic.fit(X_train, y_train).score(X_test, y_test))
    elif(method == 'RF'):
        rt = RandomForestClassifier(n_estimators = method_featureValue)
        print('Random Forest score: %f'
              % rt.fit(X_train, y_train).score(X_test, y_test))
        
    end = time.time()
    print("Runtime in seconds:")
    print(end - start)
    
    '''
    print("Prediction Vector")
    print logistic.predict(X_test[0:50])
    print("Intended Output Vector")
    print y_test[0:50]
'''
    
# -----------------------------------------------------------BEGIN HERE -------------------------------------------------------
''' 
filepath = "C:\Users\staslist\Desktop\UCIClassWork\CS284A\Project\cullpdb+profile_6133.npy.gz"

raw_dataset = np.load(filepath)
dataset = np.reshape(raw_dataset, (6133, 700, 57))
'''

'''

# CREATE A DECIMAL REPRESENTATION DATASET
dataset_decimal = np.empty((6133, 700, 28))
index_1 = 0
index_2 = 0
while index_1 < 6133:
    while index_2 < 700:
        dataset_decimal[index_1][index_2][0] = one_hot_to_decimal_amino_acid_sequence(dataset[index_1][index_2][0:22])
        dataset_decimal[index_1][index_2][1] = one_hot_to_decimal_secondary_structure(dataset[index_1][index_2][22:31])
        dataset_decimal[index_1][index_2][2:28] = dataset[index_1][index_2][31:57]
        index_2 += 1
    index_2 = 0
    index_1 += 1
print dataset_decimal.shape
print dataset.shape

print dataset_decimal[0, 0, :]
print dataset_decimal[0, 0, 0]
print dataset_decimal[0, 0, 1]

np.save('cullpdb+profile_6133_3D_decimal.npy', dataset_decimal)

'''

filepath = "C:\Users\staslist\Desktop\UCIClassWork\CS284A\Project\cullpdb+profile_6133_3D_decimal.npy"
dataset_decimal = np.load(filepath)
#print dataset_decimal.shape

#print dataset_decimal[0, 0, :]
#print dataset_decimal[0, :, 1]

'''
Original Dataset description
1st dimension: N proteins (6133 proteins in total)

2nd dimension: Amino Acids / Secondary Structures (a sequence of 700 amino acid labels / secondary structure labels)

3rd dimesions: Features (57 in total) [0, 21] Amino Acid residues;
[22, 30] Secondary Structure Labels; [31, 32] N- and C- terminals
[33, 34] Relative and absolute Solvent Accessibility, used only for training.
[35, 56] Sequence profile.

For a fixed protein & fixed amino acid the 57 features describe that particular
amino acid / secondary structure.

The first 30 features are either 0s or 1s. They indicate that given amino
acid & secondary structure labels apply to the fixed amino acid / secondary
structure of the fixed protein. Only one amino acid label should be 1 and the rest should be 0s.
Only one secondary structure label should be 1 and the rest should be 0s.
The remaining features take on a range of a values for any given amino acid /
secondary structure.

Example: the dataset[0, 0, :] represents the feature description of the 1st amino acid / secondary
structure of the 1st protein in the database.

print dataset[0, 0, :]

For a fixed protein & fixed feature the 700 amino acid labels describe the
distribution of that particular feature in the amino acid / secondary structure sequence.

Example: the dataset[0, :, 0] indicates which amino acids in the sequence have label A
Example: the dataset[0, :, 21] indicates which amino acids in the sequence have label noSequence
Example: the dataset[0, :, 22] indicates which secondary structures in the sequence have label L

print dataset[0, :, 21]
'''

'''
The dataset_decimal uses decimal (instead of one hot) representation for amino acid
and secondary structure labels. Dataset_decimal.shape is (6133, 700, 28) rather than
the original shape of (6133, 700, 57).
'''
#++++++++++++++++++GOAL: predict the secondary structure sequence using the amino acid sequence.++++++++++++++++++++++

#---------APPROACH(1) USE AMINO ACID SEQUENCE ONLY + ONLY 1 to 1 MAPPING of AMINO ACIDS TO SECONDARY STRUCTURE LABELS------------

'''
start = time.time()

# N is number of training data samples
N = 100

# M is number of testing data samples
M = 272

# Remove all features except the amino acid label.
X_train = dataset_decimal[0:N, :, 0:1]
X_test = dataset_decimal[5605:5877, :, 0:1]

y_train = dataset_decimal[0:N, :, 1]
y_test = dataset_decimal[5605:5877, :, 1]

# Change the X_train into a 2D matrix & the y_train in 1D matrix

X_train = np.reshape(X_train, ((N * 700), 1))
y_train = np.reshape(y_train, (N * 700))

# REMOVE NON SEQUENCE VALUES FROM TESTING DATA & RESHAPE the X_test matrix in 2D & the y_test matrix in 1D
cutoff = endOfSequenceIndex(X_test[0, :, 0], 21)
X_temp = X_test[0, 0:cutoff, :]
y_temp = y_test[0, 0:cutoff]
index_0 = 1
# index through 272 test proteins
while index_0 < M:
    cutoff = endOfSequenceIndex(X_test[index_0, :, 0], 21)
    X_temp = np.concatenate((X_temp, X_test[index_0, 0:cutoff, :]), axis = 0)
    y_temp = np.concatenate((y_temp, y_test[index_0, 0:cutoff]), axis = 0)
    index_0 += 1

X_test = X_temp
y_test = y_temp


print("AMINO ACID SEQUENCE ONLY WITH 1 to 1 MAPPING of AMINO ACIDS TO SECONDARY STRUCTURE LABELS")
logistic = linear_model.LogisticRegression()
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))

end = time.time()
print("Runtime in seconds:")
print(end - start)

#print("Prediction Vector")
#print logistic.predict(X_test[0:50])
#print("Intended Output Vector")
#print y_test[0:50]
'''




#-----APPROACH(2) USE AMINO ACID SEQUENCE & OTHER FEATURES + ONLY 1 to 1 MAPPING of AMINO ACIDS TO SECONDARY STRUCTURE LABELS------

'''
start = time.time()

# N is number of training data samples
N = 100
# M is number of testing data samples
M = 272

# Remove the secondary structure label from the input data.
X_train = np.delete(dataset_decimal[0:N, :, :], 1, 2)
X_test = np.delete(dataset_decimal[5605:5877, :, :], 1, 2)

y_train = dataset_decimal[0:N, :, 1]
y_test = dataset_decimal[5605:5877, :, 1]

# Change the X_train into a 2D matrix & the y_train in 1D matrix

X_train = np.reshape(X_train, ((N * 700), 27))
y_train = np.reshape(y_train, (N * 700))

# REMOVE NON SEQUENCE VALUES FROM TESTING DATA & RESHAPE the X_test matrix in 2D & the y_test matrix in 1D
cutoff = endOfSequenceIndex(X_test[0, :, 0], 21)
X_temp = X_test[0, 0:cutoff, :]
y_temp = y_test[0, 0:cutoff]
index_0 = 1
while index_0 < M:
    cutoff = endOfSequenceIndex(X_test[index_0, :, 0], 21)
    X_temp = np.concatenate((X_temp, X_test[index_0, 0:cutoff, :]), axis = 0)
    y_temp = np.concatenate((y_temp, y_test[index_0, 0:cutoff]), axis = 0)
    index_0 += 1

X_test = X_temp
y_test = y_temp

print("USE AMINO ACID SEQUENCE & OTHER FEATURES + ONLY 1 to 1 MAPPING of AMINO ACIDS TO SECONDARY STRUCTURE LABELS")

logistic = linear_model.LogisticRegression()
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))

#print("Prediction Vector")
#print logistic.predict(X_test[0:50])
#print("Intended Output Vector")
#print y_test[0:50]

end = time.time()
print("Runtime in seconds:")
print(end - start)

'''
#-----------------------------------------APPROACH(3) USE AMINO ACID SEQUENCES OF LENGTH (K)----------------------------------

'''
print "SINGLE FEATURE LR METHODS WITH N = 100 C = 0.1"
classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'LR', 0.1)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'LR', 0.1)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'LR', 0.1)

print "SINGLE FEATURE LR METHODS WITH N = 100 C = 1.0"
classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'LR', 1.0)

print "SINGLE FEATURE LR METHODS WITH N = 100 C = 10"
classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'LR', 10)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'LR', 10)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'LR', 10)

print "SINGLE FEATURE LR METHODS WITH N = 100 C = 100"
classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'LR', 100)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'LR', 100)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'LR', 100)
'''

'''

print "SINGLE FEATURE LR METHODS WITH N = 10 C = 1.0"
classify_KSEQ(dataset_decimal, 1, 10, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 3, 10, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 17, 10, 272, False, 'LR', 1.0)

print "SINGLE FEATURE LR METHODS WITH N = 100 C = 1.0"
classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'LR', 1.0)

print "SINGLE FEATURE LR METHODS WITH N = 1000 C = 1.0"
classify_KSEQ(dataset_decimal, 1, 1000, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 3, 1000, 272, False, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 17, 1000, 272, False, 'LR', 1.0)
'''

'''

print "SINGLE FEATURE RANDOM FOREST METHODS WITH N = 100 N_ESTIMATORS = 10"

classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'RF', 10)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'RF', 10)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'RF', 10)

print "SINGLE FEATURE RANDOM FOREST METHODS WITH N = 100 N_ESTIMATORS = 100"

classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'RF', 100)

print "SINGLE FEATURE RANDOM FOREST METHODS WITH N = 100 N_ESTIMATORS = 1000"

classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'RF', 1000)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'RF', 1000)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'RF', 1000)

print "SINGLE FEATURE RANDOM FOREST METHODS WITH N = 10 N_ESTIMATORS = 100"

classify_KSEQ(dataset_decimal, 1, 10, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 3, 10, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 17, 10, 272, False, 'RF', 100)

print "SINGLE FEATURE RANDOM FOREST METHODS WITH N = 100 N_ESTIMATORS = 100"

classify_KSEQ(dataset_decimal, 1, 100, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 3, 100, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 17, 100, 272, False, 'RF', 100)

print "SINGLE FEATURE RANDOM FOREST METHODS WITH N = 1000 N_ESTIMATORS = 100"

classify_KSEQ(dataset_decimal, 1, 1000, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 3, 1000, 272, False, 'RF', 100)

classify_KSEQ(dataset_decimal, 17, 1000, 272, False, 'RF', 100)

'''

#-----------------------------------------APPROACH(4) USE AMINO ACID SEQUENCES OF LENGTH (K) & Additional Features----------------------------------

#print "MULTI FEATURE LR METHODS"

#classify_KSEQ(dataset_decimal, 1, 100, 272, True, 'LR', 1.0)


print "MULTI FEATURE LR METHODS WITH N = 100 C = 0.1"

classify_KSEQ(dataset_decimal, 3, 100, 272, True, 'LR', 0.1)

print "MULTI FEATURE LR METHODS WITH N = 100 C = 1.0"

classify_KSEQ(dataset_decimal, 3, 100, 272, True, 'LR', 1.0)

print "MULTI FEATURE LR METHODS WITH N = 100 C = 10.0"

classify_KSEQ(dataset_decimal, 3, 100, 272, True, 'LR', 10.0)

print "MULTI FEATURE LR METHODS WITH N = 100 C = 100.0"

classify_KSEQ(dataset_decimal, 3, 100, 272, True, 'LR', 100.0)
'''

classify_KSEQ(dataset_decimal, 17, 100, 272, True, 'LR', 1.0)

classify_KSEQ(dataset_decimal, 17, 500, 272, True, 'LR', 5.0)


print "MULTI FEATURE RANDOM FOREST METHODS"

classify_KSEQ(dataset_decimal, 1, 100, 272, True, 'RF', 100)

classify_KSEQ(dataset_decimal, 3, 100, 272, True, 'RF', 100)

classify_KSEQ(dataset_decimal, 17, 100, 272, True, 'RF', 100)

classify_KSEQ(dataset_decimal, 17, 500, 272, True, 'RF', 100)
'''
