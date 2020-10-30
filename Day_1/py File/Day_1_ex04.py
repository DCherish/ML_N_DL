#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x = tf.placeholder(tf.float32)


# In[3]:


y = tf.placeholder(tf.float32)


# In[4]:


z = tf.multiply(x, y)


# In[5]:


sess = tf.Session()


# In[6]:


print(sess.run(z, feed_dict = {x: 3., y: 5.}))


# In[ ]:




