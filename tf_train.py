# -*- coding: utf-8 -*-
import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
import input_data
import time
import tf_possion

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,80000, 0.8, staircase=True)

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
y_predict,rmse,grad_x_rmse,grad_y_rmse,mean_db,f_obj= tf_possion.model()
train_step = tf.train.AdamOptimizer(learning_rate).minimize(f_obj,global_step=global_step)
saver=tf.train.Saver(max_to_keep=5)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logs', sess.graph)

sess.run(tf.global_variables_initializer())
x_train,d_train,y_train = input_data.input_data(test=False)
x_test,d_test,y_test = input_data.input_data(test=True)

epochs = 1000
train_size = x_train.shape[0]
global batch
batch = 50
test_size = x_test.shape[0]

train_index = list(range(x_train.shape[0]))
test_index = list(range(x_test.shape[0]))

find_handle=open('train record.txt', mode='a')
find_handle.write('===============================================================\n')
for i in range(epochs):

    random.shuffle(train_index)
    random.shuffle(test_index)
    x_train,d_train,y_train = x_train[train_index],d_train[train_index],y_train[train_index]
    x_test,d_test,y_test = x_test[test_index],d_test[test_index],y_test[test_index]

    for j in range(0,train_size,batch):
        train_step.run(feed_dict={tf_possion.x:x_train[j:j+batch],tf_possion.d:d_train[j:j+batch],tf_possion.y_actual:y_train[j:j+batch],tf_possion.keep_prob:0.5})

    temp_loss = 0
    train_loss=0
    train_db=0
    temp_db=0
    for j in range(0,train_size,batch):
        train_loss = rmse.eval(feed_dict={tf_possion.x:x_train[j:j+batch],tf_possion.d:d_train[j:j+batch],tf_possion.y_actual:y_train[j:j+batch],tf_possion.keep_prob: 1.0})
        temp_loss = temp_loss + train_loss
        train_db = mean_db.eval(feed_dict={tf_possion.x:x_train[j:j+batch],tf_possion.d:d_train[j:j+batch],tf_possion.y_actual:y_train[j:j+batch],tf_possion.keep_prob: 1.0})
        temp_db = temp_db + train_db
    train_loss = temp_loss/(train_size/batch)
    train_db = temp_db/(train_size/batch)

    temp_loss = 0
    temp_grad_x = 0
    temp_db = 0
    gradx_result = 0
    test_db = 0
    test_loss = 0
    for j in range(0,test_size,batch):
        test_loss = rmse.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.d:d_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        test_db = mean_db.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.d:d_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        gradx_result = grad_x_rmse.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.d:d_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
        temp_loss = temp_loss+test_loss
        temp_db = temp_db+test_db
        temp_grad_x = temp_grad_x + gradx_result

    test_loss = temp_loss/(test_size/batch)
    test_db = temp_db/(test_size/batch)
    gradx_result = temp_grad_x/(test_size/batch)
    if i==1000:
        for j in range(0,test_size,batch):
            y_print = y_predict.eval(feed_dict={tf_possion.x:x_test[j:j+batch],tf_possion.d:d_test[j:j+batch],tf_possion.y_actual:y_test[j:j+batch],tf_possion.keep_prob: 1.0})
            sio.savemat('./1000test/'+'1000testall'+str(j)+'.mat',{'true_potential':y_test[j:j+batch],'predict_potential':y_print,'perm':x_test[j:j+batch],'distance':d_test[j:j+batch]})
        for j in range(0,train_size,batch):
            y_print = y_predict.eval(feed_dict={tf_possion.x:x_train[j:j+batch],tf_possion.d:d_train[j:j+batch],tf_possion.y_actual:y_train[j:j+batch],tf_possion.keep_prob: 1.0})
            sio.savemat('./1000train/'+'1000trainall'+str(j)+'.mat',{'true_potential':y_train[j:j+batch],'predict_potential':y_print,'perm':x_train[j:j+batch],'distance':d_train[j:j+batch]})

    summary_str = sess.run(merged_summary_op,feed_dict={tf_possion.x:x_test[0:batch],tf_possion.d:d_test[0:batch],tf_possion.y_actual:y_test[0:batch],tf_possion.keep_prob: 1.0})
    summary_writer.add_summary(summary_str,i)
    saver.save(sess, './checkpoint_dir/MyModel', global_step=i)
    print ('epoch {0} done! train_loss:{1} test_loss:{2} grad_x:{3} train_db:{4} test_db:{5} global_step:{6} learning rate:{7}'.format(i,train_loss, test_loss,gradx_result,train_db,test_db,global_step.eval(),learning_rate.eval()))
    find_handle.write('epoch {0} done! train_loss:{1} test_loss:{2} grad_x:{3} train_db:{4} test_db:{5} global_step:{6} learning rate:{7} \n'.format(i,train_loss, test_loss,gradx_result,train_db,test_db,global_step.eval(),learning_rate.eval()))
find_handle.close()
