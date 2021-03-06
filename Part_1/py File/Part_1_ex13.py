#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.set_random_seed(777)


# In[3]:


x_data = [1, 2, 3]
y_data = [1, 2, 3]


# In[4]:


W = tf.Variable(tf.random_normal([1]), name = "weight")


# In[5]:


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[6]:


hypothesis = X * W


# In[7]:


cost = tf.reduce_mean(tf.square(hypothesis - Y))


# In[8]:


learning_rate = 0.1


# In[9]:


gradient = tf.reduce_mean((W * X - Y) * X)


# In[10]:


descent = W - learning_rate * gradient


# In[11]:


update = W.assign(descent)


# In[13]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(21):
        _, cost_val, W_val = sess.run(
            [update, cost, W], feed_dict = {X: x_data, Y: y_data})
        print(step, cost_val, W_val)


# In[ ]:




