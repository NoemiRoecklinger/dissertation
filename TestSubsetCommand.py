
# coding: utf-8

# # This is my file to create all features for the ML process
# 
# I have my cleaned file with revised classes (1-11): Dataset_for_ML 
# 
# Now I take that and calculate all the other features for it 

# In[1]:


import pandas as pd
import os
import numpy as np
import sklearn 
import scipy.linalg as scplinag
from sklearn.neighbors import KDTree


# In[2]:


def calcCovarianceMatrix(data):
    """
    Function to compute the covariance matrix.
    
    Input: Dataset of 3D points; i.e. array of dimension: #points x 3 
    Output: 3x3 covariance matrix (np.array)
    """
    # Create covariance matrix and array to store the mean values for x_mean, y_mean, z_mean
    C = np.zeros((data.shape[1], data.shape[1]))
    mean_xyz = []
    # Calculate all mean values
    for i in range(0, dataxyz.shape[1]):
        mean_xyz.append(dataxyz[:,i].mean())
    mean_xyz = np.array(mean_xyz)
    # Check whether dimensions agree 
    if dataxyz[:,0].size != dataxyz[:,1].size or dataxyz[:,0].size != dataxyz[:,2].size:
        print "X, Y and Z must be of same dimensions."
    else:
        # For each row in covariance matrix C
        for i in range(0, C.shape[0]):
            # For each column in covariance matrix C
            for j in range(0, C.shape[1]):
                C[i,j] = 0
                # For each point in the dataset, access x, y, z-values
                for point in dataxyz:
                    # For each point, access x,y and z in all combinations (xx, xy, xz, yx, yy, yz etc)
                    C[i][j] = C[i][j] + (point[i]-mean_xyz[i])*(point[j]-mean_xyz[j])
    # Divide by the total number of points                
    C = (1.0/dataxyz.shape[0]) * C
    return C 


# In[3]:


# Get eight parameters for each point

def calcFeatureDescr(covarianceMatrix):
    """
    Function to compute the 8 feature descriptors for each point.
    
    Input: 3x3 Covariance matrix of a point and its neighbourhood 
    
    Output: np Array with feature descriptors as described by Weinmann et al. (1D array with 8 elements)
    
    """
    D, V = scplinag.eigh(C)
    # We sort the array with eigenvalues by size (from smallest to largest value)
    D.sort()
    # Get eigenvectors
    e1 = V[2] # eigenvector in direction of largest variance
    e2 = V[1] # second eigenvector, perpend. to e1
    e3 = V[0]
    # Find the eigenvalues
    evalue1 = D[2] # largest
    evalue2 = D[1]
    evalue3 = D[0] # smallest

    # Linearity
    lambda1 = (evalue1 - evalue2) / evalue1
    # Planarity
    lambda2 = (evalue2 - evalue3) / evalue1
    # Scattering
    lambda3 = evalue3 / evalue1
    # Omnivariance
    lambda4 = pow(evalue1*evalue2*evalue3, 1/3.0)
    # Anisotropy
    lambda5 = (evalue1 - evalue3) / evalue1
    # Eigentropy
    s = 0
    for elem in D:
        s = s + (elem*np.log(elem))
    lambda6 = (-1)*s  
    # Sum of eigenvalues
    lambda7 = sum(D)
    # Change of curvature
    lambda8 = evalue3/sum(D) 
    
    featureDescriptor = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8])
    return featureDescriptor


# In[4]:


# Define a data frame with all my data# Define  
FILE_PATH = r"../DATA"
FILE_NAME = r"/Dataset_for_ML_verticality.txt"
IMAGE_FILE_PATH = r"images"
df = pd.read_csv(FILE_PATH+FILE_NAME, delimiter=',')
df.head()


# ## Convert to NumPy array 
# 

# In[6]:


# Data is the whole dataset but as a numpy array 
data = df.values
rows, columns = data.shape
print "Number of rows:", rows
print "Number of columns", columns


# In[21]:


# Get only XYZ values
dataxyz = data[:,0:3]


# In[22]:


subset_test_ifPCbreaksDown = dataxyz[:10000]


# ## Compute all features
# 
# 

# In[ ]:


# For all points now 
# Create kd-tree
kdt = KDTree(subset_test_ifPCbreaksDown, leaf_size=40, metric='euclidean')
# Get list with indices, the first value is always the point itself
idx_list = kdt.query(subset_test_ifPCbreaksDown, k=50, return_distance=False)
store = []
for j in range(0, subset_test_ifPCbreaksDown.shape[0]):
    # Look at all points now
    neighbourhood = []
    for i in idx_list[j]:
        neighbourhood.append(subset_test_ifPCbreaksDown[i])
    neighbourhood = np.array(neighbourhood)
    # Everything we did before with dataset, we do now with the neighbourhood only
    C = calcCovarianceMatrix(neighbourhood)
    feat = calcFeatureDescr(C)
    row_with_additional_col = np.append(subset_test_ifPCbreaksDown[j], feat)
    store.append(row_with_additional_col)
store = np.array(store)
print "This is the shape of the file:", store.shape


# Create a data frame with the calculated features 
df2 = pd.DataFrame({
    'X': store[:,0],
    'Y': store[:,1],
    'Z': store[:,2],
    'lambda1': store[:,3],
    'lambda2': store[:,4],
    'lambda3': store[:,5],
    'lambda4': store[:,6],
    'lambda5': store[:,7],
    'lambda6': store[:,8],
    'lambda7': store[:,9],
    'lambda8': store[:,10]
})
df2.to_csv(FILE_PATH+'/subset_if_PC_breaksdown_features.txt', index= False)

