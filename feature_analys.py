
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


# Create np_array form the Feature-Object.
file = "auto_inception_v3_Complete_Scans_renamed_rescaled_1024_01_857566.p"
with open(file, 'rb') as f:
    feat_dict = pickle.load(f) 
image_count = len(feat_dict["paths"])
print("Image Count",image_count)
feat_dict.keys()


# In[3]:


#Extract the features
layer = []
#List of all feature layers
listof_layers = []
#Name of the pictures
image_labels = []
#All 11 layers in on single Array
all_layers = []
#can set manuelly for debugging purpose or else it get image_count
image_count_manuell = image_count
for y in range(11):
    for i in range(image_count_manuell):
        layer.append(feat_dict["features"][i][y][0])
    listof_layers.append(layer)
    layer = []
feature_extraction = []
#Gets the name of the pictures from the dictionary and delete the path
#Concat all layers in one layer
leer = []
pattern = "/.*/"
for i in range(image_count_manuell):
    image_labels.append(re.sub(pattern," ",str(feat_dict["paths"][i])))
    for u in range(11):
        leer = np.concatenate((leer,feat_dict["features"][i][u][0]))  
    feature_extraction.append(leer)
    leer = []  


# In[4]:


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


# In[5]:


list_of_distances = []
print("Distance Matrix function beginn...")
t_start_pdist = time()
for i in range(11):
    #Pairwise distances between observations in n-dimensional space.
    list_of_distances.append(sc.distance.pdist(matrix_list[i],'euclidean'))
t_stop_pdist = time()


# In[6]:


t_start_pdist_one = time()
distance_matrix = sc.distance.pdist(matrix[0],"euclidean")
t_stop_pdist_one = time()


# In[15]:


t_start_linkage_one = time()
D = hc.linkage(distance_matrix)
t_stop_linkage_one = time()

hc.dendrogram(D,
              truncate_mode = "lastp",
              leaf_font_size = 9,
              show_contracted=True
             )
plt.title("Dendogram from whole Distancematrix")
plt.xlabel("Clustersize")
plt.ylabel("Distance")
plt.savefig('dendogram_from_10048.jpg')
plt.show()


# In[8]:


#cutree = cluster.hierarchy.cut_tree(X, n_clusters=[5,5,10])
clusters_one =  hc.fcluster(D,4,criterion='maxclust')
#assign the clusters the the data
cluster_output_one = pd.DataFrame({'pictures':image_labels , 'cluster':clusters_one})
#save the dataframe as csv
cluster_output_one.to_csv('cluster_output_one.csv')


# In[23]:


list_of_linkages = []
t_start_linkage = time()
for i in range(11):
    list_of_linkages.append(hc.linkage(list_of_distances[i], method ="complete"))
t_stop_linkage = time()

list_of_linkages[0]


# In[24]:


hc.dendrogram(list_of_linkages[0], 
              truncate_mode = "lastp",
              leaf_font_size = 9,
              show_contracted=True,
              #labels = image_labels
             
             )
plt.title("Dendogram from parts of Distancematrixes")
plt.xlabel("Clustersize")
plt.ylabel("Distnace")

#plt.axhline(y=300)
plt.savefig('dendogram_0.jpg')
plt.show()


# In[21]:


#cutree = cluster.hierarchy.cut_tree(X, n_clusters=[5,5,10])
clusters =  hc.fcluster(list_of_linkages[0],4,criterion='maxclust')
#assign the clusters the the data
cluster_output = pd.DataFrame({'pictures':image_labels , 'cluster':clusters})
#save the dataframe as csv
cluster_output.to_csv('cluster_0.csv')


# In[13]:


print ('the function pdist takes %f' %(t_stop_pdist-t_start_pdist))
print ('the function pdist_one takes %f' %(t_stop_pdist_one-t_start_pdist_one))
print ('the function linkage takes %f' %(t_stop_linkage - t_start_linkage))
print ('the function linkage_one takes %f' %(t_stop_linkage_one - t_start_linkage_one))


# In[14]:


print("Saving:  ")
np.save("distance_matrix.npy", distance_matrix)
for i in range(11):
    np.save("distance_matrix_new"+ str(i) +".npy", list_of_distances[i])  
print("Saved !   ")

