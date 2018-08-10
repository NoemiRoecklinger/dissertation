
# coding: utf-8

# # Evaluate the classifier
# 
# In file 7_ I created the classifiers for the initial experiment setup. 
# Now I need to evaluate them 

# In[2]:


import numpy as np
import sklearn 
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
import pandas as pd
from sklearn.externals import joblib


# In[3]:


NN = 10


# In[13]:


# Uploadng my datasets
# Define a data frame with all my data# Define  
FILE_PATH = r"../DATA/ML_datasets/Initial_setup"
META_FILE_PATH = "../DATA/META"
IMAGE_FILE_PATH = r"images"

# Training data
y_train = np.loadtxt(FILE_PATH+'/y_train_'+ str(NN) +'NN.txt', delimiter=',')
X_train = np.loadtxt(FILE_PATH+'/X_train_'+ str(NN) +'NN.txt', delimiter=',')

# Testing data
y_test = np.loadtxt(FILE_PATH+'/y_test_'+ str(NN) +'NN.txt', delimiter=',')
X_test = np.loadtxt(FILE_PATH+'/X_test_'+ str(NN) +'NN.txt', delimiter=',')


# In[14]:


# For RF classifier: cast as float32 
y_train = y_train.astype('float32')
X_train = X_train.astype('float32')
y_test = y_test.astype('float32')
X_test = X_test.astype('float32')


# In[10]:


###uploading the saved classifier
filename = '/randomforest_model_'+ str(NN) +'NN.sav'
clf = joblib.load(FILE_PATH + filename)


# ## Make predictions
# 
# Now predict values for the whole training set (on data the model has seen before)

# In[19]:


predictions_train = clf.predict(X_train)
predictions_test = clf.predict(X_test)

np.savetxt(FILE_PATH+'/predictions_train_'+ str(NN) +'NN.txt', predictions_train, delimiter=',')
np.savetxt(FILE_PATH+'/predictions_test_'+ str(NN) +'NN.txt', predictions_test, delimiter=',')

print "This is the number of NN:", NN


# ## Now evaluate the model

# In[17]:


score_train = clf.score(X_train, y_train)
print "Score on training dataset:", score_train


# In[18]:


score_test = clf.score(X_test, y_test)
print "Score on testing dataset:", score_test

