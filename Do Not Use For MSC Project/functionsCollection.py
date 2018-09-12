# Functions I will need for my dissertation

import pandas as pd
import os
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
import scipy.linalg as scplinag
from sklearn.neighbors import KDTree

# This will be needed for when I read in a dataset and I want to extract x,y,z and label  
def getColumns(data, get_label = True):
    """
    INPUT: 
    data: dataset as np.array with 3 (xyz) or 4 (xyz label) attributes
    get_label: Boolean, whether or not to extract the label as well 
    OUTPUT: 
    xyz (optional: label) as numpy.array with dim: size of dataset x 1
    """
    # Get xyz values as column vectors 
    x = subset[:,0]
    y = subset[:,1]
    z = subset[:,2]
    
    if get_label == True:
        x = subset[:,0]
        y = subset[:,1]
        z = subset[:,2]
        label = subset[:,3]
        return x,y,z,label
    else: 
        return x,y,z

# To get the covariance matrix of my point cloud 		
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
	
	
# Function to get the number of x % of a dataset 
def getXpercentOfDataset(dataset, percent):
	"""
	INPUT: full dataset, does not matter how many attributes, as long as np array and %
	OUTPUT: 
	m = number of e.g. 20% of the dataset (if 'percent' = 20)
	n = complimentrary number, i.e. 80% (in this case) 
	"""
	m = (dataset.shape[0]*percent)/100.0
	n = (dataset.shape[0]*(100-percent))/100.0
	return int(m), int(n)

	
# Get the accuracy in % for a classification result
# My function to just count the number of times I got it right 
def getAccuracy(predictions, labels):
    """
    INPUT: predicted values of a classifier (test set); true labels of that test data set
    OUTPUT: accuracy in %
    """
    count = 0
    for i in range(0, labels.shape[0]):
        if predictions[i] == labels[i]:
            # each time the prediction matches with the actual label, the counter goes up 1 
            count = count + 1
    count = float(count)
    acc = (count/labels.shape[0])*100
    
    return acc