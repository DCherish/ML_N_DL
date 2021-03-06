#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]


# In[3]:


W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)


# In[4]:


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


# In[5]:


hypothesis = x * W + b


# In[6]:


cost = tf.reduce_mean(tf.square(hypothesis - y))


# In[7]:


train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# In[8]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1000):
        sess.run(train, {x: x_train, y: y_train})
        
    W_val, b_val, cost_val = sess.run([W, b, cost],
                                      feed_dict = {x: x_train, y: y_train})
        
    print(f"W: {W_val} b: {b_val} cost: {cost_val}")


# In[ ]:




