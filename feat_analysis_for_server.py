
# coding: utf-8

# ## Feature Extraction
# 
# #### The extracted features are going to be clustered
# 
# ###### First we load the Features and convert the into summ of 11 one dimensional arrays and also 11 different array with different length of arrays  (1, 256) (1, 288) (1, 288) (1, 768) (1, 768) (1, 768) (1, 768) (1, 768) (1, 1280) (1, 2048) (1, 2048) and  also (1, 10048)

# In[1]:


import keras
import pickle
from time import time
import os
import re
import pandas as pd
import numpy as np
import scipy.spatial as sc
import scipy.cluster.hierarchy as hc
from scipy import cluster

from matplotlib.mlab import PCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering


# In[2]:


import mycluster as cl


# In[3]:


# Create np_array form the Feature-Object.
t_start_loadScans = time()
file = "auto_inception_v3_Complete_Scans_renamed_rescaled_1024_01_857566.p"
with open(file, 'rb') as f:
    feat_dict = pickle.load(f) 
t_stop_loadScans =time()
image_count = len(feat_dict["paths"])
print("Image Count",image_count)
feat_dict.keys()


# In[4]:


#Extract the features
layer = []
#List of all feature layers
listof_layers = []
#Name of the pictures
image_labels = []
image_names  = []
#All 11 layers in on single Array
all_layers = []
#can set manuelly for debugging purpose or else it get "image_count"!!!!
image_count_manuell = 3000
for y in range(11):
    for i in range(image_count_manuell):
        layer.append(feat_dict["features"][i][y][0])
    listof_layers.append(layer)
    layer = []
feature_extraction = []
#Gets the name of the pictures from the dictionary and delete the path
#Gets the label from the pictures
#Concat all layers in one layer
leer = []
pattern = "/.*/"
for i in range(image_count_manuell):
    image_names.append(re.sub(pattern," ",str(feat_dict["paths"][i])))  
    image_labels.append(feat_dict["labels"][i])
    for u in range(11):
        leer = np.concatenate((leer,feat_dict["features"][i][u][0]))  
    feature_extraction.append(leer)
    leer = []  


# In[5]:


#Make a Matrix of 48800(number of pictures) * 10048
matrix = []
shape = list(feature_extraction[0].shape)
shape[:0] =  [len(feature_extraction)]
matrix.append(np.concatenate(feature_extraction).reshape(shape))

#Convert numpy Arrays into numpy Matrix
matrix_list = []
for i in range(11):
    shape = list(listof_layers[i][0].shape)
    shape[:0] = [len(listof_layers[1])]
    matrix_list.append(np.concatenate(listof_layers[i]).reshape(shape))
    shape = 0

#for i in range(11):print(matrix_list[i].shape)


# ##### Berechnung der Matrix Distance und speichert das Ergebnis local

# In[6]:


###Pdist function berechnet die Matrix Distance aus

#list of 11 distances 256,288,288,768,768,786...
list_of_distances = []
print("Distance Matrix_list function beginn...")
t_start_pdist = time()
for i in range(11):
    #Pairwise distances between observations in n-dimensional space.
    list_of_distances.append(sc.distance.pdist(matrix_list[i],'euclidean'))
t_stop_pdist = time()

#one distance matrix 10048
print("Distance Matrix function beginn...")
t_start_pdist_one = time()
distance_matrix = sc.distance.pdist(matrix[0],"euclidean")
t_stop_pdist_one = time()

#save the distances
print("Saving:  ")
np.save("distance_matrix.npy", distance_matrix)
for i in range(11):
    np.save("distance_matrix_new"+ str(i) +".npy", list_of_distances[i])  
print("Saved !   ")


# ##### Use the Linkage function for Cluster Analysis
# 
# #####Perform Hierarchical / Agglomerative clustering

# In[7]:


list_of_linkages = []
print("Linkage_list function beginn...")
#list of 11 linkages 256,288,288,786,786...
t_start_linkage = time()
for i in range(11):
    list_of_linkages.append(hc.linkage(list_of_distances[i], method ="complete"))
t_stop_linkage = time()
#one linkage 10048
print("Linkage_one function beginn...")
t_start_linkage_one = time()
D = hc.linkage(distance_matrix)
t_stop_linkage_one = time()

#save linkages
print("Saving list of linkages:  ")
for i in range(11):
    np.savetxt("list_linkages"+ str(i) +".npy", list_of_linkages[i])  
np.savetxt("linkage_one.npy", D)


# In[8]:


print("Beginn to cluster_list and save...")
###cluster the 11 lists 256,288,288,786,786,786...
t_cluster_start_list = time()
for i in range(11):
    #cutree = cluster.hierarchy.cut_tree(X, n_clusters=[5,5,10])
    clusters = (hc.fcluster(list_of_linkages[i],2,criterion='maxclust'))
    #assign the clusters the the data
    cluster_output = pd.DataFrame({'pictures':image_names , 'cluster':clusters, "labels": image_labels})
    #save the dataframe as csv
    cluster_output.to_csv('cluster_'+str(i)+'.csv')
    clusters = 0
t_cluster_l_stop_list = time()


# In[9]:


print("Beginn to cluster_one and save...")
t_cluster_start_one = time()
#cutree = cluster.hierarchy.cut_tree(X, n_clusters=[5,5,10])
clusters_one =  hc.fcluster(D,4,criterion='maxclust')
#assign the clusters the the data
cluster_output_one = pd.DataFrame({'pictures':image_names , 'cluster':clusters_one , "labels": image_labels})
#save the dataframe as csv
cluster_output_one.to_csv('cluster_output_one.csv')
t_cluster_stop_one = time()


# In[10]:


#Check the Time: How long does the function take?

print("Loading the scanned features take %f" %(t_stop_loadScans - t_start_loadScans ))

print ('the function pdist takes %f' %(t_stop_pdist-t_start_pdist))
print ('the function pdist_one takes %f' %(t_stop_pdist_one-t_start_pdist_one))
print ('the function linkage takes %f' %(t_stop_linkage - t_start_linkage))
print ('the function linkage_one takes %f' %(t_stop_linkage_one - t_start_linkage_one))
print ("the function fcluster one takes %f" %(t_cluster_stop_one -t_cluster_start_one))


# In[11]:


cl.fancy_dendrogram(D, p=30,truncate_mode = "lastp",leaf_font_size = 9,leaf_rotation=90.,
                    show_contracted=True)

plt.show()


# In[51]:


last = D[-30:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)
# 2nd derivative of the distances
acceleration = np.diff(last, 2)  
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.xlim(2, 6)

plt.show()
# if idx 0 is the max of this we want 2 clusters
k = acceleration_rev.argmax() + 2  
print ("clusters:", k)


# In[52]:


cluster_one = hc.fcluster(D, 3, depth=10)

