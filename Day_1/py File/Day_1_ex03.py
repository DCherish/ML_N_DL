#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x = tf.Variable(tf.random_normal([784, 200], stddev = 0.35))


# In[3]:


y = tf.Variable(x.initialized_value() + 3.)


# In[4]:


init_op = tf.global_variables_initializer()


# In[5]:


sess = tf.Session()


# In[6]:


sess.run(init_op)


# In[7]:


print(sess.run(y))


# In[8]:


print(y.get_shape())


# In[ ]:




