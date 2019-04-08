
# coding: utf-8

# In[40]:


# Philip Tenteromano
# Machine Learning
# CISC 5800
# 4/7/2018
# hw3.py

# using jupyter notebook
# question 6 comment at the bottom


# In[41]:


import numpy as np


# #### Helper Functions

# In[42]:


# sigmoid helper function
def _g(h):
    return 1/(1+ np.e ** -h)


# ## Neural Network

# In[43]:


# Question 1

# Neural Net function -> 3 layers
# outputs response from layer 3 (the output layer) => a single number
# x ==> the input values, shape [1,3]
# layerNW ==> an array of weights for each layer, N
def feedForward(x, layer1W, layer2W, layer3W):
    # init output for layers
    lay1Out = np.zeros((2,))
    lay2Out = np.zeros((2,))
    lay3Out = 0
    
    # -- LAYER 1 --
    # layer 1 -- use input values to feed into layer1
    for i,v in enumerate(x):
        lay1Out[0] += layer1W[0,i] * v
        lay1Out[1] += layer1W[1,i] * v
    
    # layer 1 -- add b values
    lay1Out[0] += layer1W[0,-1]
    lay1Out[1] += layer1W[1,-1]
    
    # layer 1 -- finally, pass to sigmoid
    lay1Out[0] = _g(lay1Out[0])
    lay1Out[1] = _g(lay1Out[1])
    
    # -- LAYER 2 --
    # layer 2 -- use layer1 output to feed into layer2
    for i,v in enumerate(lay1Out):
        lay2Out[0] += layer2W[0,i] * v
        lay2Out[1] += layer2W[1,i] * v
    
    # add b values to layer 2
    lay2Out[0] += layer2W[0,-1]
    lay2Out[1] += layer2W[1,-1]

    # layer 2 -- finally, pass to sigmoid
    lay2Out[0] = _g(lay2Out[0])
    lay2Out[1] = _g(lay2Out[1])
    
    # -- LAYER 3 --
    # layer 3 -- use input values to feed into layer1
    for i,v in enumerate(lay2Out):
        lay3Out += layer3W[i] * v
    
    # layer 3 -- add b values
    lay3Out += layer3W[-1]
    
    # layer 3 -- finally, pass to sigmoid
    lay3Out = _g(lay3Out)
    
    # return the final output
    return lay3Out


# In[44]:


# feed forward

# from homework example - for question one
# x = [1, 2, 3]
# lay1w=np.array([[2, 1, 0, 1],[ 0, 2, 1, 0]])
# lay2w=np.array([[0, -2, 0],[ -1, 0, 0]])
# lay3w=np.array([1,-1,0])

# output should be 0.4624 


# In[45]:


# testQ1 = feedForward(x, lay1w, lay2w, lay3w)
# print(round(testQ1,4))


# ## Recurrent Neural Network

# In[46]:


# Question 2

# Neural Net, recurrent function -> 3 layers
# outputs response from layer 3 (the output layer) => a single number
# x ==> the input values, shape [t,3]
# layerNW ==> an array of weights for each layer, N
def feedForwardRecurrent(x, layer1W, layer2W, layer3W):
    
    # init the array of t return values
    t = x.shape[0]
    results = np.zeros((t,))
    
    # number of times to iterate
    passes = 0
    
    # init outputs
    lay1Out = np.zeros((2,))
    lay2Out = np.zeros((2,))
    lay3Out = 0
    
    # init pass outputs (lay3 will be in results array)
    lay1PastOut = np.zeros((2,))
    lay2PastOut = np.zeros((2,))
    
    while passes < t:
        # get current input
        currentInput = x[:,passes]
        
        # -- LAYER 1 --
        # layer 1 -- use input values to feed into layer1
        for i,v in enumerate(currentInput):
            lay1Out[0] += layer1W[0,i] * v
            lay1Out[1] += layer1W[1,i] * v

        # layer 1 -- add b values
        lay1Out[0] += layer1W[0,-1]
        lay1Out[1] += layer1W[1,-1]
        
        # layer 1 -- add the RECURRENT term
        lay1Out[0] += layer1W[0][-2] * lay1PastOut[0]
        lay1Out[1] += layer1W[1][-2] * lay1PastOut[1]
        
        # layer 1 -- finally, pass to sigmoid
        lay1Out[0] = _g(lay1Out[0])
        lay1Out[1] = _g(lay1Out[1])

        # -- LAYER 2 --
        # layer 2 -- use layer1 output to feed into layer2
        for i,v in enumerate(lay1Out):
            lay2Out[0] += layer2W[0,i] * v
            lay2Out[1] += layer2W[1,i] * v

        # add b values to layer 2
        lay2Out[0] += layer2W[0,-1]
        lay2Out[1] += layer2W[1,-1]
        
        # layer 2 -- add the RECURRENT term
        lay2Out[0] += layer2W[0][-2] * lay2PastOut[0]
        lay2Out[1] += layer2W[1][-2] * lay2PastOut[1]

        # layer 2 -- finally, pass to sigmoid
        lay2Out[0] = _g(lay2Out[0])
        lay2Out[1] = _g(lay2Out[1])

        # -- LAYER 3 --
        # layer 3 -- use input values to feed into layer1
        for i,v in enumerate(lay2Out):
            lay3Out += layer3W[i] * v

        # layer 3 -- add b values
        lay3Out += layer3W[-1]
        
        # layer 3 -- add the RECURRENT term
        if passes > 0:
            lay3Out += layer3W[-2] * results[passes-1]

        # layer 3 -- finally, pass to sigmoid
        lay3Out = _g(lay3Out)
        
        # store output into results
        results[passes] = lay3Out
        
        # store output results to look back on
        lay1PastOut = np.copy(lay1Out)
        lay2PastOut = np.copy(lay2Out)
        
        # reset the current outputs arrays to 0 (operations rely on += )
        lay1Out.fill(0)
        lay2Out.fill(0)
        lay3Out = 0
        
        # increment and repeat
        passes += 1
     
    return results


# In[47]:


# recurent feed word

# homework example - for question 2
# xMat=np.array([[1, 2, 3],[2, 0, 1],[ 3, 1, 0]])
# lay1w=np.array([[2, 1, 0, -2, 1],[ 0, 2, 1, 0, 0]])
# lay2w=np.array([[0, -2, -1, 0],[ -1, 0, 1, 0]])
# lay3w=np.array([1,-1,3,0])

# output should be 0.4624, 0.7724, 0.891 (t # of numbers)


# In[48]:


# testQ2 = feedForwardRecurrent(xMat, lay1w, lay2w, lay3w)

# for i in testQ2:
#     print(round(i,4))


# ## Start Part 2: questions 3-6

# ### Take this out on submission - ALL IMPORTS AND PLOTS

# In[49]:


# import matplotlib.pyplot as plt
# %matplotlib inline


# ### Uncomment this if you want to import data

# In[50]:


# import csv
# reader = csv.reader(open("data/mnist_train.csv", "rt", encoding="utf8"), delimiter=",")
# x = list(reader)
# digits = np.array(x).astype('float')

# import scipy.io as sio

# loadedData = sio.loadmat('data/hw3NNfactors.mat')
# nnFactors = loadedData['nnFactors']


# ### Show an example digit

# In[51]:


# n = 12
# digitX = digits[n,1:].reshape((28,28))
# plt.imshow(digitX)


# ### Question3 

# In[52]:


# Question 3

# Finding Weights 3
# single data point x, and components u, find the weights z
# x ==> the input pixel values, shape [1,784]
# uMat ==> matrix of the components, shape [784, 40]
# returns shape [1,40] array of z weights
def findWeights3(x, uMat):
    # init the weights array, shape [1,40]
    zWeights = np.zeros((1,40))
    # get x as vector
    x = x.reshape(784,)
    
    # loop over every component, u
    for i,u in enumerate(uMat.T):
        # dot product the component with the data point
        zWeights[0][i] = np.dot(u,x)
    
    return zWeights


# In[53]:


# test Q3

# n = 0
# xi = digits[n,1:]
# testQ3 = findWeights3(xi, nnFactors)

# print(testQ3.shape)
# print(testQ3)


# # Question 4

# In[54]:


# Question 4

# Finding Weights 4
# single data point x, and components u, find the weights z
# x ==> the input pixel values, shape [1,784]
# uMat ==> matrix of the components, shape [784, 40]
# returns shape [1,40] array of z weights
def findWeights4(x, uMat):
    
    # init the weights array, shape [1,40]
    zWeights = np.zeros((1,40))
    # get x as vector
    x = x.reshape(784,)
    
    # list of removed components
    removedComponents = []
    
    # conditional compoenents
    iters = 0;
    highest_zq = 1
    # component to remove
    # highest_index = -1
    
    while iters < 40 and highest_zq > 0:
        
        # loop over every component, u
        for i,u in enumerate(uMat.T):
            # check if component is removed
            if i not in removedComponents:
                # dot product the component with the data point
                zWeights[0][i] = np.dot(u,x)
            
        # get the highest weight from the (u @ x) dot product
        highest_zq = np.amax(zWeights[0])
        
        # store q - the specifc u, z index for removal
        q = np.argmax(zWeights[0])
        
        # update x by removing uq (x <- x – zq uq)
        x = x - (zWeights[0][q] * uMat.T[q])
        
        # take these weights and components out
        zWeights[0][q] = 0
        removedComponents.append(q);
        
        iters += 1
        
    return zWeights


# In[55]:


# xi = digits[n,1:]
# testQ4 = findWeights4(xi, nnFactors)

# print(testQ4.shape)
# print(testQ4)


# ### Plotting digit

# In[56]:


# n = 4
# xi = digits[n,1:]
# z = findWeights4(xi, nnFactors)

# test = np.zeros((784,))

# for i, u in enumerate(nnFactors.T):
#     test += z[0][i] * u
# plt.imshow(test.reshape(28,28))


# In[57]:


# Question 5

# build new x from components and their weights
def xEstimate(z, uMat):
    xi = np.zeros((784,))
    
    for i, u in enumerate(uMat.T):
        xi += z[0][i] * u
    
    xi = xi.reshape(784,1)
    return xi


# ### Start question 6 

# In[58]:


# findWeights3 - getting average error on entire dataset

# numPoints = digits.shape[0]
# total = 0.
# avgMSE3 = 0.

# for x in digits[:,1:]:
#     zWeights = findWeights3(x, nnFactors)
#     newX = xEstimate(zWeights, nnFactors)
#     newX = newX.reshape(784,)

#     mse = sum((x - newX) ** 2)
#     total += mse

# avgMSE3 = total / numPoints
# print(avgMSE3)


# In[59]:


# findWeights4 - getting average error on entire dataset

# numPoints = digits.shape[0]
# total4 = 0.
# avgMSE4 = 0.

# for x in digits[:,1:]:
#     zWeights = findWeights4(x, nnFactors)
#     newX = xEstimate(zWeights, nnFactors)
#     newX = newX.reshape(784,)

#     mse = sum((x - newX) ** 2)
#     total4 += mse

# avgMSE4 = total4 / numPoints
# print(avgMSE4)


# In[60]:


# findWeights3 has much larger error than findWeights4
# avgMSE3 > avgMSE4


# In[61]:


# the difference in error between the algorithms
# print(avgMSE3 - avgMSE4)


# In[62]:


# findWeights4 is more than 3x better than findWeights3
# avgMSE3/avgMSE4


# In[63]:


# - QUESTION 6 - DESCRIPTION OF ERRORS 

# As we can see above, the first method, findWeights3()
# tries to use every weight/component to reconstruct the x point
# Unfortunately, this results in a much higher avereage MSE over the dataset
#
# For findWeights4(), we can see that, by subtracting larger weights
# from x, we begin to bring other weights closer to 0, eventually below 0.
# This ends up being more effective, lowering the average MSE over the dataset
# by a factor of over 3
