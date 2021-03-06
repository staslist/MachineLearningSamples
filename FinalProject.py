# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 04:42:18 2017

@author: staslist
"""

# used the following tutorial to begin using keras library
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# DNN tips http://rishy.github.io/ml/2017/01/05/how-to-train-your-dnn/

# I have attempted to use tensorflow-gpu, but could not set it up on Windows 10 
# in a reasonable amount of time. This has limited the size of the network that 
# I could train in a reasonable amount of time on my personal machine. 

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.client import device_lib
from sklearn.model_selection import GridSearchCV

print(device_lib.list_local_devices())

np.random.seed(0)

# Data Loading
X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)

# The test data
Xte = np.genfromtxt('data/X_test.txt', delimiter=None)

Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xtr, Ytr = ml.shuffleData(Xtr, Ytr)

# Taking a subsample of the data so that trains faster.
Xt, Yt = Xtr[:10000], Ytr[:10000] 

XtS, params = ml.rescale(Xt)
XvS, _ = ml.rescale(Xva, params)
XteS, _ = ml.rescale(Xte, params)

# Settled on some initial variables such epochs=700, batch_size=1000, 
# loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'accuracy'
# and activation = 'relu' via references found online & trial/error

scores = []
num_hidden_layers = [1, 5, 10, 30, 50, 100]

for i in num_hidden_layers:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14, activation='relu'))
    model.add(Dropout(0.2))
    j = 0
    while j < i:
        model.add(Dense(14, activation='relu'))
        model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=1000)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)
# without regularization ~10 hidden layers seem to deliver best results (as opposed to 1, 5, 30, 50...) 
# with dropout regularization ~10 hidden layers still delivers best results


scores = []
num_hidden_neurons = [7, 14, 28, 56, 112, 224, 448]
num_hidden_layers = 10

for i in num_hidden_neurons:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(14, activation='relu'))
    model.add(Dropout(0.2))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(i, activation='relu'))
        model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=1000)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)


# without regularization num hidden neurons = 14 performed the best (~70%)
# most publications/textbooks on the topic suggest using number of neurons in hidden layers 
# that is in between number of neurons in the input layer and number of neurons in 
# the output layer
# the number of neurons in input layer is generally = # of features in the dataset

# with dropout regularization num hidden neurons between 28 & 112 (inclusive) performed the best (~70%)
# note: doubling number of neurons per hidden layer roughly doubles the runtime

scores = []
activations = ['relu', 'selu', 'elu', 'tanh', 'sigmoid']
num_hidden_layers = 10
num_hidden_neurons = 14

for acti in activations:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation=acti))
    #model.add(Dropout(0.2))
    model.add(Dense(14, activation=acti))
    #model.add(Dropout(0.2))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(num_hidden_neurons, activation=acti))
        #model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=1000)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)

# elu(70.4%) > tanh(70.26%) > selu > relu > sigmoid(68.9%). However, the differences were small.

scores = []
losses = ['mean_squared_logarithmic_error', 'mean_absolute_error', 'binary_crossentropy']
num_hidden_layers = 10
num_hidden_neurons = 14

for lo in losses:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation='elu'))
    #model.add(Dropout(0.2))
    model.add(Dense(14, activation='elu'))
    #model.add(Dropout(0.2))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(num_hidden_neurons, activation='elu'))
        #model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss=lo, optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=1000)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)

# as expected binary cross entropy delivers best results

scores = []
rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# can vary learning rate for RMSprop and SGD
optimizers = [sgd, rms, 'Adagrad', 'Adadelta', 'Adam' ,'Adamax', 'Nadam']
num_hidden_layers = 10
num_hidden_neurons = 14

for opti in optimizers:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation='elu'))
    #model.add(Dropout(0.2))
    model.add(Dense(14, activation='elu'))
    #model.add(Dropout(0.2))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(num_hidden_neurons, activation='elu'))
        #model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=1000)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)

# very similar results with all of the optimizers. Adamax performed slightly better than all 
# other optimizers

scores = []
batch_sizes = [200, 500, 1000, 2000]
num_hidden_layers = 10
num_hidden_neurons = 14

for b_s in batch_sizes:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation='elu'))
    #model.add(Dropout(0.2))
    model.add(Dense(14, activation='elu'))
    #model.add(Dropout(0.2))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(num_hidden_neurons, activation='elu'))
        #model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=b_s)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)

# very similar results, batch size of 500 seems to work best


scores = []
num_epochs = [100, 250, 500, 750]
num_hidden_layers = 10
num_hidden_neurons = 14

for n_e in num_epochs:
    model = Sequential()
    
    model.add(Dense(14, input_dim=14, activation='elu'))
    #model.add(Dropout(0.2))
    model.add(Dense(14, activation='elu'))
    #model.add(Dropout(0.2))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(num_hidden_neurons, activation='elu'))
        #model.add(Dropout(0.2))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    # Fit the model
    model.fit(XtS, Yt, epochs=n_e, verbose=0, batch_size=500)
    # evaluate the model
    score = model.evaluate(XvS, Yva)
    # append accuracy only
    scores.append(score[1]*100)
print(scores)

# No noticeable improvement between 100 & 500 epochs. 

# The hypermarameters can be tuned much further. However, due to time constraints I decided 
# to stop here. 
'''
num_hidden_layers = 10;
num_hidden_neurons = 14;

model = Sequential()
    
model.add(Dense(14, input_dim=14, activation='elu'))
model.add(Dense(14, activation='elu'))
j = 0
while j < num_hidden_layers:
    model.add(Dense(num_hidden_neurons, activation='elu'))
    j = j + 1
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
# Fit the model
model.fit(XtS, Yt, epochs=100, verbose=0, batch_size=500)
# evaluate the model
score = model.evaluate(XvS, Yva)
print(score)
'''
# If time try selecting (there are noisy / not relevant features) or adding features
# (there is interdependence between features)
# Can safely assume from the problem write up that all of the features are relevant
'''
XtSP = ml.transforms.fpoly(XtS, degree=2, bias=False)
XvSP = ml.transforms.fpoly(XvS, degree=2, bias=False)
XteSP = ml.transforms.fpoly(XteS, degree=2, bias=False)
# there are now 119 features: original 14 features + 14 square features + additional cross features such as x1*x2
# Rescale the data matrix so that the features have similar ranges / variance
XtSP, params = ml.transforms.rescale(XtSP)
XvSP, _ = ml.transforms.rescale(XvSP, params)
XteS, _ = ml.transforms.rescale(XvSP, params)
'''
'''
num_hidden_layers = 10;
num_hidden_neurons = 119;

model = Sequential()
    
model.add(Dense(119, input_dim=119, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(119, activation='elu'))
model.add(Dropout(0.2))
j = 0
while j < num_hidden_layers:
    model.add(Dense(num_hidden_neurons, activation='elu'))
    model.add(Dropout(0.2))
    j = j + 1
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
# Fit the model
model.fit(XtSP, Yt, epochs=100, batch_size=500)
# evaluate the model
score = model.evaluate(XvSP, Yva)
print(score)
'''

'''
# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, neurons = 119):
    # create model
    num_hidden_layers = 10;
    num_hidden_neurons = neurons;
    
    model = Sequential()
        
    model.add(Dense(119, input_dim=119, activation='elu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(119, activation='elu'))
    model.add(Dropout(dropout_rate))
    j = 0
    while j < num_hidden_layers:
        model.add(Dense(num_hidden_neurons, activation='elu'))
        model.add(Dropout(dropout_rate))
        j = j + 1
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [500, 1000, 2000]
epochs = [100, 400, 700]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
dropout_rate = [0, 0.15, 0.35, 0.5]
neurons = [40, 80, 119]
param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate, neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
print('Began automated hyperparameter grid search.')
grid_result = grid.fit(XtSP, Yt)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''


'''
# calculate predictions
predictions = model.predict(XteS)
print(predictions)

Yte = np.vstack((np.arange(Xte.shape[0]), predictions[:, 0])).T
np.savetxt('data/Y_submit.txt', Yte, '%d, %.2f', header = 'ID,Prob1', comments='', delimiter=',')
'''