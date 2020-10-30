#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.set_random_seed(777)


# In[3]:


x_train = [1, 2, 3]
y_train = [1, 2, 3]


# In[4]:


W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")


# In[5]:


hypothesis = x_train * W + b


# In[6]:


cost = tf.reduce_mean(tf.square(hypothesis - y_train))


# In[7]:


train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# In[8]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])
        
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)


# In[ ]:




