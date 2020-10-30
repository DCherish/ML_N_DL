#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x = tf.Variable(2.)


# In[3]:


print(x)


# In[4]:


init_op = tf.global_variables_initializer()


# In[5]:


sess = tf.Session()


# In[6]:


sess.run(init_op)


# In[7]:


print(sess.run(x))


# In[ ]:




