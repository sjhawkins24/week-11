import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from time import time

#Load diamonds dataset 
data = sns.load_dataset("diamonds")
#Get the numerical columns 
num_data = data.iloc[:,[0, 4, 5, 6, 7, 8, 9] ]
 

def kmeans(X, k):
    """Wraper function for kmeans"""
    #define the kmeans model 
    kmeans = KMeans(n_clusters = k)
    #fit the model
    kmeans.fit(X)

    #Get the labels and centroids 
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    res_tuple = (centroids, labels)
    return(res_tuple)

def kmeans_diamonds(n, k):
    "Function to run kmeans on diamonds data"
    #Load diamonds dataset 
    #data = sns.load_dataset("diamonds")
    #Get the numerical columns 
    #num_data = pd.DataFrame(data.iloc[:,[0, 4, 5, 6, 7, 8, 9] ])
 
    #Get the subset of the data
    subset = pd.DataFrame(num_data.iloc[0:(n), :])
    
    return(kmeans(subset, k))

def kmeans_timer(n, k, n_iter = 5):
    """Function to check the timing of the kmeans_diamond function"""
    t = []
    for n in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        t.append(time() - start)
    return(t)