#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


x = tf.constant(3)


# In[3]:


print(x)


# In[4]:


sess = tf.Session()


# In[5]:


result = sess.run(x)


# In[6]:


print(result)


# In[ ]:




