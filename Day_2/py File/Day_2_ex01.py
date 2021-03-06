#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.set_random_seed(777)


# In[3]:


x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]


# In[4]:


X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])


# In[5]:


W = tf.Variable(tf.random_normal([2, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))


# In[6]:


hypothesis = tf.sigmoid(tf.matmul(X, W) + b)


# In[7]:


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))


# In[8]:


train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# In[9]:


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# In[12]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)
            
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                      feed_dict={X: x_data, Y: y_data})
    
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


# In[ ]:




