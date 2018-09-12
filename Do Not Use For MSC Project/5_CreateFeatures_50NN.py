
# coding: utf-8

# # This is my file to create all features for the ML process
# 
# I have my cleaned file with revised classes (1-11): Dataset_for_ML 
# 
# Now I take that and calculate all the other features for it 

# In[3]:


import pandas as pd
import os
import numpy as np
import sklearn 
import scipy.linalg as scplinag
from sklearn.neighbors import KDTree
import time
from scipy.spatial import distance


# In[4]:


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
    for i in range(0, data.shape[1]):
        mean_xyz.append(data[:,i].mean())
    mean_xyz = np.array(mean_xyz)
    # Check whether dimensions agree 
    if data[:,0].size != data[:,1].size or data[:,0].size != data[:,2].size:
        print "X, Y and Z must be of same dimensions."
    else:
        # For each row in covariance matrix C
        for i in range(0, C.shape[0]):
            # For each column in covariance matrix C
            for j in range(0, C.shape[1]):
                C[i,j] = 0
                # For each point in the dataset, access x, y, z-values
                for point in data:
                    # For each point, access x,y and z in all combinations (xx, xy, xz, yx, yy, yz etc)
                    C[i][j] = C[i][j] + (point[i]-mean_xyz[i])*(point[j]-mean_xyz[j])
    # Divide by the total number of points                
    C = (1.0/data.shape[0]) * C
    return C 


# In[5]:


# Get eight parameters for each point

def calcFeatureDescr(covarianceMatrix):
    """
    Function to compute the 8 feature descriptors for each point.
    
    Input: 3x3 Covariance matrix of a point and its neighbourhood 
    
    Output: np Array with feature descriptors as described by Weinmann et al. (1D array with 8 elements)
    
    """
    D, V = scplinag.eig(covarianceMatrix)
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
    misc1 = np.prod(D)
    lambda4 = pow(misc1,(1.0/3))
    # Anisotropy
    lambda5 = (evalue1 - evalue3) / evalue1
    # Eigentropy
    s = 0
    count = 0
    for elem in D:
        if elem == 0:
            s = 0
            count = 1
        else:
            # Only if bigger than 0
            misc2 = (elem*np.log(elem))
            if misc2 == 0:
                print "Multiplication result too close to zero."
                s = 0
            else:
                s = s + misc2
    lambda6 = (-1)*s  
    # Sum of eigenvalues
    lambda7 = sum(D)
    # Change of curvature
    lambda8 = evalue3/sum(D)
    
    featureDescriptor = np.array([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8])
    return featureDescriptor, count


# In[6]:


# Get local point density

def calcPointDensity(number_NN, radius):
    """
    Function to compute the local point density as introduced by Weinmann 2013 (Formula 2).
    
    Input: number of NN (scalar), calculated radius (scalar) of the neighbourhood determined
            by the number of points. Ideally, this should have a small STD as density is not varying too
            much after cleaning the dataset 
    
    Output: Local point density (scalar)
    
    """
    
    D = (number_NN+1.0)/((4.0/3)*np.pi*pow(radius, 3))
    
    return D


# In[7]:


# Define a data frame with all my data# Define  
FILE_PATH = r"../DATA"
FILE_NAME = r"/Dataset_for_ML_verticality.txt"
IMAGE_FILE_PATH = r"images"
df = pd.read_csv(FILE_PATH+FILE_NAME, delimiter=',')


# ## Convert to NumPy array 
# 

# In[8]:


# Data is the whole dataset but as a numpy array 
data = df.values
rows, columns = data.shape
print "Number of rows:", rows
print "Number of columns", columns


# In[9]:


# Get only XYZ values
dataxyz = data[:,0:3]


# ## Compute all features
# 
# 

# In[23]:


NN = 50
# For all points now 
start = time.time()
# Create kd-tree
kdt = KDTree(dataxyz, leaf_size=40, metric='euclidean')
print 'Created tree in:', float(time.time()-start)/60, 'minutes'
# Get list with indices, the first value is always the point itself
start = time.time()
idx_list = kdt.query(dataxyz, k=NN, return_distance=False, sort_results = True)
print 'Queried tree in:', float(time.time()-start)/60, 'minutes'
start = time.time()
store = []
radii = []
point_density = []
counter = []
for j in range(0, dataxyz.shape[0]):
    # Look at all points now
    neighbourhood = []
    for i in idx_list[j]:
        neighbourhood.append(dataxyz[i])
    neighbourhood = np.array(neighbourhood)
    # Calculate radius for neighbourhood
    idx_first_point = idx_list[j][0]
    idx_last_point = idx_list[j][-1]
    first_point = dataxyz[idx_first_point]
    last_point = dataxyz[idx_last_point]
    radius_neighbourhood = distance.euclidean(first_point, last_point)
    radii.append(radius_neighbourhood)
    # Point density 
    D = calcPointDensity(NN, radius_neighbourhood)
    point_density.append(D)
    # Everything we did before with dataset, we do now with the neighbourhood only
    C = calcCovarianceMatrix(neighbourhood)
    feat, count = calcFeatureDescr(C)
    counter.append(count)
    row_with_additional_col = np.append(dataxyz[j], feat)
    store.append(row_with_additional_col)
store_complex = np.array(store)
store = np.real(store_complex)
print "This is the shape of the file:", store.shape
print 'Computed features in:', float(time.time()-start)/60, 'minutes'
c = 0
for elem in counter:
    if elem == 1:
        c = c + 1
print "This is how many elements (eigenvalues) are 0:", c

# Create a data frame with the calculated features 
df2 = pd.DataFrame({
    'X': store[:,0],
    'Y': store[:,1],
    'Z': store[:,2],
    'relative_height': df['relative_height'],
    'verticality': df['verticality'],
    'lambda1': store[:,3],
    'lambda2': store[:,4],
    'lambda3': store[:,5],
    'lambda4': store[:,6],
    'lambda5': store[:,7],
    'lambda6': store[:,8],
    'lambda7': store[:,9],
    'lambda8': store[:,10],
    'radius_neighbourhood': radii,
    'local_density': point_density,
    'class': df['class']
})
df2.to_csv(FILE_PATH+'/5_Data_ML_attributes_50NN.txt', index= False)

