#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.set_random_seed(777)


# In[3]:


x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# In[4]:


X = tf.placeholder("float", shape=[None, 4])
Y = tf.placeholder("float", shape=[None, 3])
nb_classes = 3


# In[5]:


W = tf.Variable(tf.random_normal([4, nb_classes], name='weight'))
b = tf.Variable(tf.random_normal([nb_classes], name='bias'))


# In[6]:


hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)


# In[7]:


cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))


# In[8]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# In[9]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
        
        if step % 200 == 0:
            print(step, cost_val)
            
    print('--------------------')
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))
    
    print('--------------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))
    
    print('--------------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))
    
    print('--------------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(a, sess.run(tf.argmax(all, 1)))


# In[ ]:




