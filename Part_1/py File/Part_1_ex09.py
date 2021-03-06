#!/usr/bin/env python
# coding: utf-8

# In[1]:


for x in "banana":
    print(x)


# In[2]:


fruits = ["apple", "banana", "cherry"]


# In[3]:


for x in fruits:
    print(x)
    if x == "banana":
        break


# In[4]:


for x in range(6):
    print(x)


# In[5]:


for x in range(2, 6):
    print(x)


# In[6]:


for x in range(2, 30, 3):
    print(x)


# In[7]:


for x in range(2, 32, 3):
    print(x)


# In[8]:


for x in range(2, 33, 3):
    print(x)


# In[9]:


for i, name in enumerate(['body', 'foo', 'bar']):
    print(i, name)


# In[10]:


x = ('apple', 'banana', 'cherry')
y = enumerate(x)


# In[11]:


print(list(y))


# In[ ]:




