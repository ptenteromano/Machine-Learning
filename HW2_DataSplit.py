
# coding: utf-8

# In[12]:


# Philip Tenteromano
# 2/19/2019
# CISC 6930
# Data Mining
# Dr. Yijun Zhao
#
# Split and save data files

# run this before main file


# In[13]:


import pandas as pd


# In[14]:


df = pd.read_csv('train-1000-100.csv')


# In[15]:


# this file takes the training data above and creates 3 more training files
# using the first 50, 100, and 150 samples, respectively 


# In[16]:


df[:50].to_csv('train-50(1000)-100.csv', index_label=False)


# In[17]:


df[:100].to_csv('train-100(1000)-100.csv', index_label=False)


# In[18]:


df[:150].to_csv('train-150(1000)-100.csv', index_label=False)


# In[19]:


def produceDataFiles():
    df = pd.read_csv('train-1000-100.csv')
    
    df[:50].to_csv('train-50(1000)-100.csv', index_label=False)
    df[:100].to_csv('train-100(1000)-100.csv', index_label=False)
    df[:150].to_csv('train-150(1000)-100.csv', index_label=False)

