
import copy
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import sys
import cv2
from PIL import Image

network_dim = int(256 / 2)  # ensure that this is always greater than max node number that occurs in your data

masks = glob.glob('S:/studenten/Rausch/06_Studienarbeit/03_CNN/generate_data/data/train_less128_2000imgs/label/*')
masks = masks[0:20]
orgs = glob.glob("S:/studenten/Rausch/06_Studienarbeit/03_CNN/generate_data/data/train_less128_2000imgs/image/*.png")
orgs = orgs[0:20]
#every training image has less than 128 nodes

#training images
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    I = cv2.imread(image)
    key = image
    graph_label = np.load(masks[int(key[-14:-9])], allow_pickle=True)
    graph_label_norm = graph_label.copy()
    positions = graph_label[:, :2, 0]
    #normalize positions
    pos_norm = np.zeros(positions.shape)
    for i in range(len(positions)):
        pos_norm[i][0] = np.round((positions[i][0]/I.shape[1])*512, 0)
        pos_norm[i][1] = np.round((positions[i][1] /I.shape[0])*512, 0)
    graph_label_norm[:, :2, 0] = pos_norm
    #pad the label to obtain uniform array sizes
    #graph_label_padded = np.pad(graph_label_norm, ((0, network_dim-graph_label.shape[0]), (0, network_dim-graph_label.shape[1]), (-9.9, -9.9)))
    #graph_label_padded = np.pad(graph_label_norm, ((0, 0), (0, 0), (0, 0)))
    #graph_label_padded = graph_label_norm
    imgs_list.append(np.array(Image.open(image).convert('L').resize((512,512))))
    #print(graph_label_padded)
    #masks_list_position.append(np.array(graph_label_norm[:, 0:2, 0]))
    #masks_list_adjacency.append(np.array(graph_label_norm[:, 2:, 0]))
    masks_list.append(graph_label_norm)
imgs_np = np.asarray(imgs_list)

from keras_unet.utils_regine import plot_graph_on_img, plot_nodes_on_img
node_thick = 6
index = 8
save = True

masks_np = np.asarray(masks_list[index])
#uniform array sizes are necessary
y_positions_label = masks_np[:,0:2, 0]
y_adjacency_label = masks_np[:,2:, 0]

node_img = plot_nodes_on_img(imgs_np[index,:,:], y_positions_label, node_thick)
fig = plot_graph_on_img(imgs_np[index,:,:], y_positions_label, y_adjacency_label)

print(np.shape(y_positions_label))
print(np.shape(np.random.rand(np.shape(y_positions_label)[0],np.shape(y_positions_label)[1])))
y_positions_label = y_positions_label+3*np.random.rand(np.shape(y_positions_label)[0],np.shape(y_positions_label)[1])

print(' y_adjacency_label', y_adjacency_label[0:10,0:10])
y_adjacency_label[0,1] =  0
y_adjacency_label[1,0] =  0
y_adjacency_label[2,3]  = 0
y_adjacency_label[3,2]  = 0
y_adjacency_label[4,5]  = 0
y_adjacency_label[5,4]  = 0

node_img = plot_nodes_on_img(imgs_np[index,:,:], y_positions_label, node_thick)
fig = plot_graph_on_img(imgs_np[index,:,:], y_positions_label, y_adjacency_label)




