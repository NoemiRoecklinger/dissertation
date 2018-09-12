
# coding: utf-8

# # Create new classifiers
# 
# So i initially tried everything with the classifier as proposed by Weinmann 
# depth = 4, number = 100
# 
# Really bad results (excel spreadsheet, 58 percent accuracy for NN10)
# 
# Now after trying different thing: scaling provided worse results, like 9 percent, really bad 
# Now play around with different settings, try different scenarios 
# 
# This will only be done for NN10, and then take the best option 

# In[1]:


import numpy as np
import sklearn 
from sklearn.neighbors import KDTree
import time
from sklearn import metrics
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


# In[2]:


NN = 10


# In[3]:


# Uploadng my datasets
# Define a data frame with all my data  
FILE_PATH = r"../DATA/ML_datasets/Initial_setup"
META_FILE_PATH = "../DATA/META"
IMAGE_FILE_PATH = r"images"

# Training data
y_train = np.loadtxt(FILE_PATH+'/y_train_'+ str(NN) +'NN_3842samples.txt', delimiter=',')
X_train = np.loadtxt(FILE_PATH+'/X_train_'+ str(NN) +'NN_3842samples.txt', delimiter=',')

# Testing data
y_test = np.loadtxt(FILE_PATH+'/y_test_'+ str(NN) +'NN.txt', delimiter=',')
X_test = np.loadtxt(FILE_PATH+'/X_test_'+ str(NN) +'NN.txt', delimiter=',')


# In[4]:


# For RF classifier: cast as float32 
y_train = y_train.astype('float32')
X_train = X_train.astype('float32')
y_test = y_test.astype('float32')
X_test = X_test.astype('float32')


# ## Now train RF with many different settings 
# 
# As tried before with Dave, try different hyperparameters 

# In[13]:


results_train = []
results_test = []
# maxDepth
for md in range(4, 14, 2):
    # number of estimators (trees)
    for ne in range(100, 1100, 200):
        clf_eval = RandomForestClassifier(max_depth=md, random_state=42, n_estimators=ne, criterion='gini')
        clf_eval.fit(X_train, y_train)
        score_train = clf_eval.score(X_train, y_train)
        score_test = clf_eval.score(X_test, y_test)
        results_train.append([md, ne, score_train])
        results_test.append([md, ne, score_test])


# In[ ]:


# Save these files in Initial setup path 
np.savetxt(FILE_PATH+'/DAVE_EVALUATION_10Example_train.txt', results_train, delimiter=',')
np.savetxt(FILE_PATH+'/DAVE_EVALUATION_10Example_test.txt', results_test, delimiter=',')


# In[14]:


print 'Run succesfully'

