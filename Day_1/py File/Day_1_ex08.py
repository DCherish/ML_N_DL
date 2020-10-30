#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


# In[3]:


row_r1 = a[1, :]


# In[4]:


row_r2 = a[1:2, :]


# In[5]:


print(row_r1, row_r1.shape)


# In[7]:


print(row_r2, row_r2.shape)


# In[8]:


col_r1 = a[:, 1]


# In[9]:


col_r2 = a[:, 1:2]


# In[10]:


print(col_r1, col_r1.shape)


# In[11]:


print(col_r2, col_r2.shape)


# In[ ]:




