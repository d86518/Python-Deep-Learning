#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 23:55:06 2018

@author: Vince
"""
### Case 1:
# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier # https://keras.io/scikit-learn-api/ (You can use Sequential Keras models (single-input only) as part of your Scikit-Learn workflow via the wrappers found at keras.wrappers.scikit_learn.py.)

help(KerasClassifier) # Implementation of the scikit-learn classifier API for Keras.

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
     
# load dataset
dataset = numpy.loadtxt('/Users/Vince/cstsouMac/Python/Examples/DeepLearning/py_codes/data/pima-indians-diabetes.csv', delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = KerasClassifier(build_fn=create_model, verbose=0) # built_fn argument

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs) # use dict to make a param grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4)
# python并行（1）https://blog.csdn.net/mmc2015/article/details/51835190

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
import time
start = time.time()
grid_result = grid.fit(X, Y)
end = time.time()
print(end - start) # 165.65647315979004 (n_jobs=-1) -> 50.45935416221619 (n_jobs=4)

type(grid_result)
dir(grid_result)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

### Case 2:
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape


len(train_labels)


train_labels


test_images.shape


len(test_labels)


test_labels


# Before training, we will preprocess our data by reshaping it into the shape that the network expects, and scaling it so that all values are in 
# the `[0, 1]` interval. Previously, our training images for instance were stored in an array of shape `(60000, 28, 28)` of type `uint8` with 
# values in the `[0, 255]` interval. We transform it into a `float32` array of shape `(60000, 28 * 28)` with values between 0 and 1.


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# We also need to categorically encode the labels, a step which we explain in chapter 3:


train_labels[0:9]



from keras.utils import to_categorical
train_labels_ohe = to_categorical(train_labels)
print(train_labels_ohe[0:1])
test_labels_ohe = to_categorical(test_labels)


nodes = [32, 64, 128, 256, 512] # number of nodes in the hidden layer
lrs = [0.001, 0.002, 0.003] # learning rate, default = 0.001
epochs = 15
batch_size = 64
from keras import optimizers

def build_model(nodes=10, lr=0.001):
    model = Sequential()
    model.add(Dense(nodes, kernel_initializer='uniform', input_dim=784, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    opt = optimizers.RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return(model)

model = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=batch_size, verbose=0) # built_fn argument, and fix epochs and batch_size 

param_grid = dict(nodes=nodes, lr=lrs)
param_grid # use dict to make a param grid

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=4, refit=True, verbose=2) # refit argument

# Code analysis based on the context, not the objects in the environment.
import time
start = time.time()
grid_result = grid.fit(X=train_images, y=train_labels_ohe)
end = time.time()
print(end - start) # 21.4min (n_jobs=1) -> 50.45935416221619 (n_jobs=4)
grid_result = grid.fit(X=train_images, y=train_labels_ohe)
#Fitting 3 folds for each of 15 candidates, totalling 45 fits
#[CV] lr=0.001, nodes=32 ..............................................
#[CV] ............................... lr=0.001, nodes=32, total=  14.9s
#[CV] lr=0.001, nodes=32 ..............................................
#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   15.4s remaining:    0.0s
#[CV] ............................... lr=0.001, nodes=32, total=  15.0s
#[CV] lr=0.001, nodes=32 ..............................................
#[CV] ............................... lr=0.001, nodes=32, total=  15.5s
#[CV] lr=0.001, nodes=64 ..............................................
#[CV] ............................... lr=0.001, nodes=64, total=  16.3s
#[CV] lr=0.001, nodes=64 ..............................................
#[CV] ............................... lr=0.001, nodes=64, total=  16.5s
#[CV] lr=0.001, nodes=64 ..............................................
#[CV] ............................... lr=0.001, nodes=64, total=  15.7s
#[CV] lr=0.001, nodes=128 .............................................
#[CV] .............................. lr=0.001, nodes=128, total=  20.1s
#[CV] lr=0.001, nodes=128 .............................................
#[CV] .............................. lr=0.001, nodes=128, total=  20.0s
#[CV] lr=0.001, nodes=128 .............................................
#[CV] .............................. lr=0.001, nodes=128, total=  20.1s
#[CV] lr=0.001, nodes=256 .............................................
#[CV] .............................. lr=0.001, nodes=256, total=  25.4s
#[CV] lr=0.001, nodes=256 .............................................
#[CV] .............................. lr=0.001, nodes=256, total=  26.7s
#[CV] lr=0.001, nodes=256 .............................................
#[CV] .............................. lr=0.001, nodes=256, total=  27.3s
#[CV] lr=0.001, nodes=512 .............................................
#[CV] .............................. lr=0.001, nodes=512, total=  43.6s
#[CV] lr=0.001, nodes=512 .............................................
#[CV] .............................. lr=0.001, nodes=512, total=  38.4s
#[CV] lr=0.001, nodes=512 .............................................
#[CV] .............................. lr=0.001, nodes=512, total=  36.3s
#[CV] lr=0.002, nodes=32 ..............................................
#[CV] ............................... lr=0.002, nodes=32, total=  14.4s
#[CV] lr=0.002, nodes=32 ..............................................
#[CV] ............................... lr=0.002, nodes=32, total=  14.5s
#[CV] lr=0.002, nodes=32 ..............................................
#[CV] ............................... lr=0.002, nodes=32, total=  14.4s
#[CV] lr=0.002, nodes=64 ..............................................
#[CV] ............................... lr=0.002, nodes=64, total=  16.1s
#[CV] lr=0.002, nodes=64 ..............................................
#[CV] ............................... lr=0.002, nodes=64, total=  16.3s
#[CV] lr=0.002, nodes=64 ..............................................
#[CV] ............................... lr=0.002, nodes=64, total=  16.4s
#[CV] lr=0.002, nodes=128 .............................................
#[CV] .............................. lr=0.002, nodes=128, total=  20.1s
#[CV] lr=0.002, nodes=128 .............................................
#[CV] .............................. lr=0.002, nodes=128, total=  20.7s
#[CV] lr=0.002, nodes=128 .............................................
#[CV] .............................. lr=0.002, nodes=128, total=  21.1s
#[CV] lr=0.002, nodes=256 .............................................
#[CV] .............................. lr=0.002, nodes=256, total=  26.2s
#[CV] lr=0.002, nodes=256 .............................................
#[CV] .............................. lr=0.002, nodes=256, total=  26.5s
#[CV] lr=0.002, nodes=256 .............................................
#[CV] .............................. lr=0.002, nodes=256, total=  26.6s
#[CV] lr=0.002, nodes=512 .............................................
#[CV] .............................. lr=0.002, nodes=512, total=  37.4s
#[CV] lr=0.002, nodes=512 .............................................
#[CV] .............................. lr=0.002, nodes=512, total=  38.2s
#[CV] lr=0.002, nodes=512 .............................................
#[CV] .............................. lr=0.002, nodes=512, total= 1.9min
#[CV] lr=0.003, nodes=32 ..............................................
#[CV] ............................... lr=0.003, nodes=32, total=  15.5s
#[CV] lr=0.003, nodes=32 ..............................................
#[CV] ............................... lr=0.003, nodes=32, total=  14.8s
#[CV] lr=0.003, nodes=32 ..............................................
#[CV] ............................... lr=0.003, nodes=32, total=  15.0s
#[CV] lr=0.003, nodes=64 ..............................................
#[CV] ............................... lr=0.003, nodes=64, total=  17.1s
#[CV] lr=0.003, nodes=64 ..............................................
#[CV] ............................... lr=0.003, nodes=64, total=  16.9s
#[CV] lr=0.003, nodes=64 ..............................................
#[CV] ............................... lr=0.003, nodes=64, total=  17.3s
#[CV] lr=0.003, nodes=128 .............................................
#[CV] .............................. lr=0.003, nodes=128, total=  21.1s
#[CV] lr=0.003, nodes=128 .............................................
#[CV] .............................. lr=0.003, nodes=128, total=  21.1s
#[CV] lr=0.003, nodes=128 .............................................
#[CV] .............................. lr=0.003, nodes=128, total=  21.1s
#[CV] lr=0.003, nodes=256 .............................................
#[CV] .............................. lr=0.003, nodes=256, total= 2.3min
#[CV] lr=0.003, nodes=256 .............................................
#[CV] .............................. lr=0.003, nodes=256, total=  27.5s
#[CV] lr=0.003, nodes=256 .............................................
#[CV] .............................. lr=0.003, nodes=256, total=  30.3s
#[CV] lr=0.003, nodes=512 .............................................
#[CV] .............................. lr=0.003, nodes=512, total=  42.1s
#[CV] lr=0.003, nodes=512 .............................................
#[CV] .............................. lr=0.003, nodes=512, total=  43.1s
#[CV] lr=0.003, nodes=512 .............................................
#[CV] .............................. lr=0.003, nodes=512, total=  40.9s
#[Parallel(n_jobs=1)]: Done  45 out of  45 | elapsed: 21.4min finished

dir(grid_result)
grid_result.n_jobs # 1

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score'] # (15,)
stds = grid_result.cv_results_['std_test_score'] # (15,)
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
import pandas as pd
pred_classes = grid.predict(test_images)
pred = pd.DataFrame({'ImageId': range(1, len(pred_classes)+1), 'Label': pred_classes})

pred.to_csv('models/mnist-nodes512-lr001.csv', index=False)

import pandas_ml as pdml
cm = pdml.ConfusionMatrix(pred_classes, test_labels)
cm.print_stats()

# Sklearn How to Save a Model Created From a Pipeline and GridSearchCV Using Joblib or Pickle? (https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli)
#type(grid) # sklearn.model_selection._search.GridSearchCV
#len(dir(grid))
#grid.best_estimator_
#
#type(grid_result) # sklearn.model_selection._search.GridSearchCV
#len(dir(grid_result))
#grid_result.best_estimator_ # same memory address as grid
#
#
#from sklearn.externals import joblib
#joblib.dump(grid_result.best_estimator_, 'models/mnist-nodes512-lr001.pkl', compress=1)
# TypeError: can't pickle _thread.RLock objects

### Reference:
# How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Tuning hyperparameters in neural network using Keras and scikit-learn https://dzubo.github.io/machine-learning/2017/05/25/increasing-model-accuracy-by-tuning-parameters.html
