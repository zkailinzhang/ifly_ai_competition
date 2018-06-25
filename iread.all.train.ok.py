#-*- coding: utf-8 -*-
# encoding=utf8


import tensorflow as tf
import tensorflow.contrib as rnn
import numpy as np 
import os
import librosa
import librosa.display
from random import shuffle
import struct
from pandas import Series,DataFrame
import pandas

import wave  # cont read pcm
import soundfile as sf   #pcm': File contains data in an unknown format.


import matplotlib.pyplot as plt


path = 'F:/1zklcode/python/speaker-recog/kaggle/dataset/data.dev'

#select one batch randomly

DIALECT = {

    'CHANGSHA':1,
    'HEBEI':2,
    'KEJIA':3,
    'MINNAN':4,
    'NANCHANG':5,
    'SHANGHAI':6

}
n_input = 13
n_steps = 250
n_hidden = 128
n_classes = 6

learning_rate = 0.001
training_iters = 100000
batch_size = 50
display_step = 10
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out' : tf.Variable(tf.random_normal([n_classes]))
}



def mfcc_batch_generator(batch_size=50):
    # maybe_download(source, DATA_DIR)
    batch_features = []
    labels = []
    sr =16000

    for i in range(batch_size):
        for root,dirs,files in os.walk(path):
            # print(dirs[0])
            # #if not files[0].endswith(".pcm"): continue
            # # print("loaded batch of %d files" % len(files))
            shuffle(dirs)
            shuffle(files)
            if not files :continue
            if not files[0].endswith(".pcm"): continue
            root = root.replace("\\","/")
            #print(root,files)
            # allpath = []
            # for file in files:
            #     print(root, file)
            #     print(os.path.join(root,file )  )
            # for allpath1 in allpath:
            #     print(allpath1 )

            #root = root.replace("\\", "/")
            #for file in files:
            splitlist = (os.path.join(root,files[0])).split('/')
            # print(splitlist,splitlist[7])
            lab = DIALECT.get(splitlist[7])
            #  print(lab)
            #if not lab :continue
            for file in files:
                if not file.endswith(".pcm"): continue
                datapcm =[]
                with open(os.path.join(root,file),'rb') as fs:
                    data = fs.read()
                    if not data: pass
                    #data =   # what 二进制？   转字符串
                    datapcm = np.frombuffer(data,dtype=np.int16)
                    #datapcm = np.fromstring(str(data), dtype=np.int16)  #偶然ValueError: string size must be a multiple of element size
                datapcm = [float(dataint) for dataint in datapcm]
                mfcc = librosa.feature.mfcc(np.array(datapcm), sr=sr,n_mfcc =13)
                #batch_features.append(mfcc)
                labels.append(lab)
                print(file,mfcc.shape,lab)

                # print(np.array(mfcc).shape)
                mfccp = np.pad(mfcc, ((0, 0), (0, 250 - len(mfcc[0]))), mode='constant', constant_values=0)
                batch_features.append(np.array(mfccp).T)
                if len(batch_features) >= batch_size:
                    yield np.array(batch_features), np.array(labels)
                    batch_features = []  # Reset for next batch
                    labels = []


                break
            break

# generator = mfcc_batch_generator()
# for i in generator:
#     print (i)


def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch = mfcc_batch_generator(batch_size)
        batch_x, batch_y = next(batch)
        # print(batch_x.shape)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y : batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss = " + \
                "{:.6f}".format(loss) + ", Training Accuracy = " + \
                "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")