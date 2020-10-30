#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a = np.array([1, 2, 3])


# In[3]:


print(a.shape)


# In[4]:


print(a[0], a[1], a[2])


# In[5]:


a[0] = 5


# In[6]:


print(a)


# In[8]:


b = np.array([[1, 2, 3], [4, 5, 6]])


# In[9]:


print(b.shape)


# In[10]:


print(b[0, 0], b[0, 1], b[1, 0])

