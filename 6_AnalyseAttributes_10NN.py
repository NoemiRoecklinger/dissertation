
# coding: utf-8

# # Machine Learning now

# In[1]:


import pandas as pd
import os
import numpy as np
import sklearn 
import scipy.linalg as scplinag
from sklearn.neighbors import KDTree
from scipy.spatial import distance


# # Whole dataset and 10 NN 
# 
# Attributes as in Weinmann, 2013, plus relative height (Weinmann, 2014) 

# In[2]:


# Define a data frame with all my data# Define  
FILE_PATH = r"../DATA"
FILE_NAME = r"/5_Data_ML_attributes_10NN.txt"
IMAGE_FILE_PATH = r"images"
df = pd.read_csv(FILE_PATH+FILE_NAME, delimiter=',')
df.rename(index=str, columns={"range": "relative_height"}, inplace = True)


# In[3]:


rows, cols = df.shape
print "Number of instances", rows
print "Number of attributes", cols


# ### I have made a mistake with the class names, therefore I need to re-name them now
# 
# In the file 3_CreateFinalClasses, I did not change the names of the classes properly. Therefore, I will have to re-name them now. 
# 
# Manually, not to confuse it.

# In[4]:


# Road (second most points)
df["class"].where(df["class"] != 2, 1, inplace=True)
# Sidewalk
df["class"].where(df["class"] != 3, 2, inplace=True)
# Curb
df["class"].where(df["class"] != 4, 3, inplace=True)
# Building (most points)
df["class"].where(df["class"] != 5, 4, inplace=True)
# Other pole like objects
df["class"].where(df["class"] != 6, 5, inplace=True)
# Small poles
df["class"].where(df["class"] != 7, 6, inplace=True)
# Pedestrians
df["class"].where(df["class"] != 11, 7, inplace=True)
# 2 wheelers
df["class"].where(df["class"] != 12, 8, inplace=True)
# 4 wheelers
df["class"].where(df["class"] != 13, 9, inplace=True)
# Trees
df["class"].where(df["class"] != 14, 10, inplace=True)
# Potted plants
df["class"].where(df["class"] != 15, 11, inplace=True)


# ## Create feature data frame with only the features I would like to have for ML and also the class

# In[5]:


df_features = df.iloc[:,3:]
df_features.drop(['radius_neighbourhood'], axis = 1, inplace = True)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


data = df_features.values
X = data[:,1:]
y = data[:,0]
print data.shape
print X.shape
print y.shape
print max(y)


# In[ ]:


split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    # Do that here
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[ ]:


print "Size X training data", X_train.shape
print "Size y training data", y_train.shape
print "Size X testing data", X_test.shape
print "Size y testing data", X_test.shape


# In[ ]:


# Save all my data as NUMPY arrays
# To load them later
# np.loadtxt()
np.savetxt(FILE_PATH+'/y_train_10NN.txt', y_train, delimiter=',')
np.savetxt(FILE_PATH+'/y_test_10NN.txt', y_test, delimiter=',')
np.savetxt(FILE_PATH+'/X_train_10NN.txt', X_train, delimiter=',')
np.savetxt(FILE_PATH+'/X_test_10NN.txt', X_test, delimiter=',')


# In[ ]:


# Compute correlation matrices to show dependencies
corr_matrix = df_features.corr()
corr_matrix.to_csv(FILE_PATH+'/corr_matrix_10NN.txt', index= True)


# In[ ]:


# Clean variables I domnt need right now

df = None
corr_matrix = None
X = None
y = None
data = None 

