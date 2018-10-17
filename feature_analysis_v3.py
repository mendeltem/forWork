#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import re
import numpy.core.multiarray
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import misc


# In[300]:


#the function changes the shape (a,b,c)->(c,b,a)
def change_shape(array):
    """changing the shape (a,b,c)->(c,b,a)
    input array of an layer 3D dimenstional
    
    returns new reshaped array
    """
    new_array = []
    for i in range(array.shape[2]):
        col = []
        for y in range(array.shape[1]):
            row = []
            for z in range(array.shape[0]):
                row.append(array[z][y][i])
            col.append(row)
        new_array.append(col)    

    return np.array(new_array)


# scaling points from orinal picture to layer matrix
#input 

def scaling_point_picture_to_layer(picture, layer_matrix, x, y):
    """scaling points from orinal picture to layer matrix
    Input:
    picture array
    layer_array
    x,y 
    Returns the scaled x,y point
    """
    original_shape = picture.shape
    print("original_shape", original_shape)
    layer_shape    = layer_matrix.shape
    print("layer_shape # 1st 3rd switched", layer_shape)
    
    width_scale = original_shape[0]/layer_shape[2]
    print("width_scale",width_scale)
    hight_scale = original_shape[1]/layer_shape[1]
    print("hight_scale",hight_scale)
    xs = int(x / w)
    ys = int(y / h)
    print("x,y",x,y)
    print("new x,y",xs,ys)
    
    return xs,ys


# In[2]:


#load the inception file 
file = "created_feature_layers/auto_inception_v3_pic_test_8a472a.p"
with open(file, 'rb') as f:
    feat_dict = pickle.load(f) 
#the number of images 
image_count = len(feat_dict["features"])
#the number of layers used 11
layer_count = len(feat_dict["features"][0])

print("Image Count: ",image_count)
print("Layer Count: ",layer_count)


# In[192]:


picture_layers = []
picture_layer  = []
image_labels   = []
image_names    = []
new = []

pattern = "/.*/"
for i in range(image_count):
    temp = []
    #get all layers and reshape every layers 
    image_names.append(re.sub(pattern," ",str(feat_dict["paths"][i])))  
    image_labels.append(feat_dict["labels"][i])
    for y in range(layer_count):
         temp.append(   
             change_shape(feat_dict["features"][i][y].reshape( feat_dict["features"][i][y][0].shape)))     
    picture_layers.append(temp)
    #get only one layer and reshape every layers 
    picture_layer.append(feat_dict["features"][i][0].reshape(feat_dict["features"][i][0][0].shape))
    
    new.append(change_shape(picture_layer[i]))
feat_dict.keys()


# In[253]:


print(np.transpose(picture_layers[0][0][1,5:7,0:3]))


print(picture_layers[0][0].shape)


# In[302]:


data = plt.imread("pic_test/3.png")
layer_shape = picture_layers[0][0].shape

print("layer",layer_shape[2])
print("whole picture : ",data.shape[1])

snipe = data[data.shape[0]-layer_shape[2]:data.shape[0],data.shape[1]-layer_shape[1]:data.shape[1]]
print("snipe         : ",snipe.shape)

data = plt.imread("pic_test/3.png")

#Eyer Movement Example
x = 1000
y = 700

new_points = scaling_point_picture_to_layer(data, picture_layers[0][0],x,y)
print(new_points)

#picture_layers[0][0][0][new_points[0]][new_points[1]]




# In[303]:


#Example of scaling
eye = snipe[new_points[1]:new_points[1]+10,new_points[0]: new_points[0]+10]
plt.imshow(eye)
plt.show()


# In[304]:


#for scaling purpose we take little part of the picture
#same size as the layer
plt.imshow(snipe)
plt.show()


# In[306]:


#whole picture
plt.imshow(data)
plt.show()

