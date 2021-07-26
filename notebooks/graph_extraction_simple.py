#%%

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import glob
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow.python.keras as keras

# your code here
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


network_dim = int(256/2)  # ensure that this is always greater than max node number that occurs in your data
path = '/home/grk/git/graph_learning/keras-unet'
os.environ['PATH'] += ':'+path
#from models_graph.custom_graph_head import custom_graph_head, custom_adj_unet
#custom_adj_unet()

from natsort import natsorted
#masks = np.load('S:/06_Studienarbeit/03_CNN/generate_data/data/train/label/adjcouput_matrix.npy',allow_pickle='TRUE').item()
masks = glob.glob('/home/grk/git/graph_learning/train_less128_2000imgs/label/*')
masks = natsorted(masks)
masks = masks[0:1990]
orgs = glob.glob("/home/grk/git/graph_learning/train_less128_2000imgs/image/*.png")
orgs = natsorted(orgs)
orgs = orgs[0:1990]


#training images
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    I = cv2.imread(image)
    key = image
    #print(key[-14:-9])
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

## Plot images + masks + overlay (mask over original)


from keras_unet.utils_regine import plot_graph_on_img, plot_nodes_on_img
node_thick = 6
index = 3
save = True

masks_np = np.asarray(masks_list[index])
#uniform array sizes are necessary
y_positions_label = masks_np[:,0:2, 0]
y_adjacency_label = masks_np[:,2:, 0]
node_img = plot_nodes_on_img(imgs_np[index,:,:], y_positions_label, node_thick)
fig = plot_graph_on_img(imgs_np[index,:,:], y_positions_label, y_adjacency_label)



#network_dim = 15
pos_vec = np.full((len(masks_list),network_dim *2), -9.9)
adj_flatten_dim = int((network_dim*network_dim-network_dim)/2)

adj_vec = np.zeros((len(masks_list),adj_flatten_dim))
for index in range(len(masks_list)):
    adj_mat = np.zeros((network_dim,network_dim))
    pos_mat = np.full((network_dim,2), -9.9)
    masks_np = np.asarray(masks_list[index])
    y_label_positions = masks_np[:,0:2, 0] # last zero --> without attributes
    y_label_adjacency = masks_np[:,2:, 0]  # last zero --> without attributes

    if y_label_positions.shape[0] >= network_dim *2:
        print('the number of labeld nodes/frame is too high for network dimension - decrease nodes in training data or consider to adapt the network size')
        pos_mat[0:network_dim,:] = y_label_positions[0:network_dim,:]
        adj_mat[0:network_dim,0:network_dim]= y_label_adjacency[0:network_dim,0:network_dim]
    else:
        #print('y_label_adjacency.shape',y_label_adjacency.shape[0])
        pos_mat[0:np.shape(y_label_positions)[0],:] = y_label_positions
        adj_mat[0:np.shape(y_label_adjacency)[0],0:np.shape(y_label_adjacency)[1]]= y_label_adjacency
  # form position matrix and adjacency in a one dimensional vector information
    pos_mat = pos_mat.reshape((network_dim*2))
    pos_vec[index,0:network_dim*2] = pos_mat

    adjacency_label_indices = np.triu_indices(network_dim, k = 1)
    adj_vec[index,0:adj_flatten_dim] = adj_mat[adjacency_label_indices]




y_label = [pos_vec, adj_vec]
print('total number of positions: ', y_label[0].shape)
print('total number of relevant adjacency entries: ', y_label[1].shape)


## Get data into correct shape, dtype and range (0.0-1.0)


print(imgs_np.max(), masks_np.max())
x = np.asarray(imgs_np, dtype=np.float32)/255
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)



#reshape
from keras_unet.utils_regine import plot_graph_on_img, plot_nodes_on_img
from models_graph.prepare_functions import create_adj_matrix, create_position_matrix
cut_off = network_dim
idx_init = 0
split_initial_img = x[idx_init,:,:,0]
split_initial_pos = y_label[0][idx_init,:]
split_initial_adj = y_label[1][idx_init,:]

split_initial_img_mat = split_initial_img
split_initial_pos_mat= create_position_matrix(split_initial_pos,cut_off)
split_initial_adj_mat = create_adj_matrix(split_initial_adj,networksize=network_dim,cut_off_size = cut_off)

#%%



node_img = plot_nodes_on_img(split_initial_img_mat ,split_initial_pos_mat,node_thick)
fig = plot_graph_on_img(split_initial_img_mat,split_initial_pos_mat, split_initial_adj_mat)


from sklearn.model_selection import train_test_split
# split data
x_train, x_val, y_train_positions,y_val_positions, y_train_adjacency, y_val_adjacency = train_test_split(x, y_label[0], y_label[1],shuffle=False, test_size=0.1, random_state=0)
x, y_label[0], y_label[1]

from models_graph.prepare_functions import convert_to_tensor
# convert from numpy to tensorflow object
x_train = convert_to_tensor(x_train)
x_val = convert_to_tensor(x_val)
y_train_positions = convert_to_tensor(y_train_positions)
y_train_adjacency = convert_to_tensor(y_train_adjacency)
y_val_positions = convert_to_tensor(y_val_positions)
y_val_adjacency = convert_to_tensor(y_val_adjacency)

## Initialize network


from models_graph.custom_graph_head import custom_graph_head, custom_adj_unet
print(x_train.shape)
input_shape = (512, 512, 1)
model = custom_adj_unet(input_size = input_shape, pretrained_weights =None, network_dim = network_dim)

#%% md

## Network shape

#%%

print('Input Shape: ',model.input_shape)
print('Output Shape: ',model.output_shape)
print('with position vector: ',model.output_shape[0], ' and adjacency vector: ', model.output_shape[1])
model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint
model_filename = 'model/graph_extract_model.h5'
model_checkpoint_callback = 'model/tmp/checkpoint'
callback_checkpoint = ModelCheckpoint(model_checkpoint_callback,
                                      save_weights_only=True,
                                      monitor='val_loss',save_freq='epoch',
                                      save_best_only='True')


## Compile + train

#
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import datetime
model.compile(
   optimizer= 'adam',
   loss={
       "pixel_position_of_nodes": keras.losses.MeanSquaredError(),
       "adjacency_matrix":   keras.losses.BinaryCrossentropy(),
   },
   loss_weights=[1, 100],
)
#print('done')
model.output


print("x_train: ", x_train.shape)
print("y_train_positions: ", y_train_positions.shape)
print(" y_train_adjacency: ", y_train_adjacency.shape)
print("x_val: ", x_val.shape)
print("y_val_positions: ", y_val_positions.shape)
print("y_val_adjacency: ", y_val_adjacency.shape)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x_train, [y_train_positions, y_train_adjacency],
          epochs = 60, batch_size=2 ,
          validation_data=(x_val,{'pixel_position_of_nodes': y_val_positions, 'adjacency_matrix': y_val_adjacency}),
          callbacks=[tensorboard_callback,callback_checkpoint],)

#%%
# Save the entire model to a HDF5 file
model.save(model_filename)

