
# coding: utf-8

# In[9]:


import numpy as np
import sklearn 
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import metrics
from sklearn.externals import joblib


# In[ ]:


# Define a data frame with all my data# Define  
FILE_PATH = r"../DATA/ML_datasets/Initial_setup"
META_FILE_PATH = "../DATA/META"
IMAGE_FILE_PATH = r"images"

y_train = np.loadtxt(FILE_PATH+'/y_train_100NN.txt', delimiter=',')
X_train = np.loadtxt(FILE_PATH+'/X_train_100NN.txt', delimiter=',')


# In[ ]:


# For RF classifier 
y_train = y_train.astype('float32')
X_train = X_train.astype('float32')


# In[ ]:


start = time.time()
clf = RandomForestClassifier(max_depth=4, random_state=42, n_estimators=100, criterion='gini')
print 'Created Random Forest in:', float(time.time()-start), 'seconds'
start = time.time()
clf.fit(X_train, y_train)
print 'Fit model in:', float(time.time()-start), 'seconds'


# In[7]:


# saving a classifier
filename = FILE_PATH+'/randomforest_model_100NN.sav'
# instead of classifier, rename based on your file
joblib.dump(clf, filename)

