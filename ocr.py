# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 21:42:59 2017

@author: shubham269
"""
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow as tf
#from __future__ import division

### image size = 25*40 pixels
num_classes = 36
image_sizer = 40  # Pixel width & height
image_sizec = 25 
pixel_depth = 255.0  # Number of levels per pixel.

batch_size = 1
num_hidden_nodes1 = 1600
num_hidden_nodes2 = 1600
num_hidden_nodes3 = 200
beta_regul = 1e-3
num_labels = num_classes

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_test_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_sizec * image_sizer))
    global_step = tf.Variable(0)

    # Variables.
    weights1 = tf.Variable(
    tf.truncated_normal(
        [image_sizec * image_sizer, num_hidden_nodes1],
        stddev=np.sqrt(2.0 / (image_sizec * image_sizer)))
    )
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
    weights2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
    weights3 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
    biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))
    weights4 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes3, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
    biases4 = tf.Variable(tf.zeros([num_labels]))
  
    # Training computation.
    lay1_train = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    lay2_train = tf.nn.relu(tf.matmul(lay1_train, weights2) + biases2)
    lay3_train = tf.nn.relu(tf.matmul(lay2_train, weights3) + biases3)
    logits = tf.matmul(lay3_train, weights4) + biases4
    
    # Predictions for the training, validation, and test data.
    test_prediction = tf.nn.softmax(logits)
    saver  = tf.train.Saver()

    
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

model_path = 'ocr_model.ckpt'

sess = tf.Session(graph=graph) 
    # Restore model weights from previously saved model
    #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, model_path)
#print("Model restored from file: %s" % model_path)


def reformat1():
    imgfile = 'char_result/'
    imglist = []

    for the_file in os.listdir(imgfile):
        imgpath=imgfile+the_file
        #print(imgpath)
        imglist.append(cv2.imread(imgpath,0)) 

    max_num_images = 50

    dataset = np.ndarray(
    shape=(max_num_images, image_sizer, image_sizec), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    image_index = 0
    for img in imglist:
        img = cv2.resize(img,(image_sizec,image_sizer))
        #print img.shape
        image_data = (img.astype(float) -pixel_depth / 2) / pixel_depth
        if image_data.shape != (image_sizer, image_sizec):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset[image_index, :, :] = image_data
        image_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    #print('Full dataset tensor:', dataset.shape)
    #print('Mean:', np.mean(dataset))
    # uncomment this if enough memory ...
    #print('Standard deviation:', np.std(dataset))
    dataset = dataset.reshape((-1, image_sizer * image_sizec)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    mapping = {0:'P',1:'I',2:'3',3:'Y',4:'N',5:'C',6:'J',7:'0',8:'Z',9:'E',10:'6',11:'Q',12:'M',13:'A',14:'V',15:'F',
    16:'W',17:'X',18:'K',19:'8',20:'D',21:'L',22:'9',23:'O',24:'U',25:'4',26:'G',27:'B',28:'5',29:'T',30:'S',
    31:'H',32:'2',33:'7',34:'R',35:'1'}

    plate_num=""
    thresh=0.06
    for l in range(dataset.shape[0]):
        test_img = [dataset[l,:]]
        feed_dict = {tf_test_dataset : test_img}
        predl = sess.run([test_prediction], feed_dict=feed_dict)
        ex_pred = np.exp(predl)
        s = np.sum(ex_pred)
        #print ex_pred
        pred = [x/s for x in ex_pred]
        ar = np.argmax(pred,2)
        ind = ar[0][0]
        #print("ind",pred[0][0][ind])
        if pred[0][0][ind]>thresh:
            plate_num=plate_num+(mapping[ind])

    #print(plate_num)
    return plate_num

#reformat1()

