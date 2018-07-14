import os

import numpy as np
import tensorflow as tf
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


files = os.listdir('/Users/xxh/Downloads/cifar-10-batches-py')
file = '/Users/xxh/Downloads/cifar-10-batches-py/data_batch_2'
data_dict = unpickle(file)
data_dict
list(data_dict.keys())

data_dict[b'batch_label']
data_dict[b'data'].shape

X = data_dict[b'data'].reshape(10000, 3, 32, 32)
X = X.transpose(0, 2, 3, 1).astype("uint8")
Y = data_dict[b'labels']
X.shape
X
sess = tf.Session()

X_ph = tf.placeholder(dtype=tf.uint8, shape=[None, 32, 32, 3])
Y_ph = tf.placeholder(dtype=tf.int64, shape=[None])

layer1 = tf.layers.conv2d(
    inputs=tf.cast(X_ph, tf.float32) / 255,  # optimizer 只在0附近优化
    filters=10,
    kernel_size=(5, 5),
    strides=(1, 1),
    padding="valid",  # same表示填补边缘保持原尺寸
    activation=tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer())

layer2 = tf.layers.conv2d(
    inputs=layer1,
    filters=20,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",  # same表示填补边缘保持原尺寸
    activation=tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer())

layer2 = tf.layers.max_pooling2d(layer2, 2, 2)
layer3 = tf.layers.conv2d(
    inputs=layer2,
    filters=10,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",  # same表示填补边缘保持原尺寸
    activation=tf.nn.relu,
    kernel_initializer=tf.contrib.layers.xavier_initializer())

layer3 = tf.layers.flatten(layer3)  # 把tensor拉直，变成样本数x特征数的矩阵

logits = tf.layers.dense(layer3, 10, activation=tf.sigmoid)  # 逻辑回归

# loss input 就是 logits， 返回一个数，表示mean loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=Y_ph, logits=logits)
# optimizer 是优化器， 对loss求梯度
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

# train_op 返回改变参数的操作， 被sess.run的时候，参数会改变
train_op = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())  # 初始化所有的kernel参数，W，b等

for ind in range(100):
    loss_value, _ = sess.run(
        [loss, train_op], feed_dict={
            X_ph: X[0:64, :, :, :],
            Y_ph: Y[0:64]
        })
    print(loss_value)
