#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


env = gym.make('FrozenLake-v0')


# In[5]:


Q = np.zeros([env.observation_space.n, env.action_space.n])


# In[6]:


learning_rate = .85
dis = .99
num_episodes = 2000


# In[7]:


rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    
    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        new_state, reward, done, _ = env.step(action)
        
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state, :]))
        state = new_state
        
        rAll += reward
        
    rList.append(rAll)
    
print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()


# In[ ]:




