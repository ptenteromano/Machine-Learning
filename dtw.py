
# coding: utf-8

# In[106]:


# Dynamic Time Warping Algorithm
from numpy import zeros as z
import numpy


# In[24]:


# dist(xi, yj)=|xi-yj| + min[ d(xi-1, yj-1), d(xi-1, yj), d(xi, yj-1) ]


# In[271]:


def dtwMatrix(X,Y):
    xAx = len(X)
    yAx = len(Y)
    
    # dist func
    dist = lambda x,y: abs(x-y)
    
    # create zero-valued matrix
    dtwMat = z((yAx,xAx))
    
    for i in range(yAx):
        for j in range(xAx):
            prev_i = i - 1
            prev_j = j - 1
            addPrev = 0
            
            if prev_i >= 0 and prev_j >= 0:
                prev1 = dtwMat[prev_i][prev_j]
                prev2 = dtwMat[prev_i][j]
                prev3 = dtwMat[i][prev_j]
                
                addPrev = min(prev1,prev2, prev3)
            elif prev_i >= 0:
                addPrev = dtwMat[prev_i][j]
            elif prev_j >= 0:
                addPrev = dtwMat[i][prev_j]
            
            dtwMat[i][j] = dist(Y[i], X[j]) + addPrev
            
    
    # swap rows so bottom left is origin
    for i in range(int(xAx/2)):
        oi = abs(7-i)
        dtwMat[[i,oi]] = dtwMat[[oi,i]]
        
    return dtwMat
            


# In[274]:


X = [39, 44, 43, 39, 46, 38, 39, 43]
Y = [37, 44, 41, 44, 39, 39, 39, 40]
m = dtwMatrix(X,Y)
m

