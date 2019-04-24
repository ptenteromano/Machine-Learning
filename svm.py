
# coding: utf-8

# In[1]:


# Philip Tenteromano
# Machine Learning
# CISC 5800
# Dr. Daniel Leeds
# 2/24/2019


# In[2]:


import scipy.io as sio
import numpy as np
import pandas as pd


# In[3]:


loadedData = sio.loadmat('hw2data.mat')
# for keys in loadedData:
#     print(keys)


# In[4]:


# store into respective variables
data1 = loadedData['data']

# data below goes together
data3 = loadedData['data3']
suppVects = loadedData['suppVec3'] 
alphas = loadedData['alphas3'].reshape(-1,)

# scalar values 
b3 = loadedData['B3'][0][0]
eps = loadedData['eps'][0][0]


# In[5]:


# Helper functions for Vector calculations
# begin with a single '_' underscore


# In[6]:


# compute dot prod if vector size is the same
def _dotProduct(a,b):
    if len(a) != len(b):
        return "Shapes mismatch"
    
    dotprod = 0
    for i in range(len(a)):
        dotprod += a[i] * b[i]
    
    return dotprod


# In[7]:


# absolute values for an entire vector 
def _absVect(vect):
    return np.array([abs(x) for x in vect])


# In[8]:


# magnitude for vector
def _magnitudeVect(vect):
    squares = 0
    for i in vect:
        squares += i ** 2
    
    return np.sqrt(squares)


# In[9]:


# Question 1

# using an absolute value kernel to create separable data

# Classifying with the formula 'sign = b + summation(alphai, yi, kernel(dataPt,vi))'
# returns the predicted label based on the separating hyperplane
def kernelClassify(dataPt, suppVecs, alphas, b): 
    sign = b
    
    # go through each support vector, store each (feature_vector, y, alpha)
    for idx, sv in enumerate(suppVecs):
        sv_Features = sv[:-1]
        sv_Yi = sv[-1]
        alph = alphas[idx]
        
        # map function - absolute values of dataPt vector, feature vector
        xi = _absVect(dataPt)
        u = _absVect(sv_Features)
        
        # kernel is just the dot product of the mapping
        kernel = _dotProduct(xi ,u)
        # find out what side of the hyperplane the dataPt is on
        sign += (alph * sv_Yi * kernel)
    
    if sign >= 1:
        return 1
    elif sign <= -1:
        return -1
    else: 
        return 0


# In[10]:


# testing on a single data point
kernelClassify(data3[2][:-1],suppVects, alphas, b3)


# In[11]:


# Question 2

# test the data points accurancy 
def testKernelClassify(dataSet, suppVecs, alphas, b, trueLabels):
    correct = 0
    numPts = len(dataSet)
    
    for idx, dataPt in enumerate(dataSet):
        predict = kernelClassify(dataPt, suppVecs, alphas, b)
        
        if predict == trueLabels[idx]:
            correct += 1
    
    return correct / numPts


# In[12]:


# store data for use in 2a and 2b below

# separate data3 into xi's and labels (part a)
testData = data3[:,:-1]
labels = data3[:,-1]

# same for support Vectors (part b)
suppVec_xVal = suppVects[:,:-1]
suppVec_labels = suppVects[:,-1]


# In[13]:


# Question 2 PART A
# Testing data3
accuracy = testKernelClassify(testData, suppVects, alphas, b3, labels)
percent = round(accuracy * 100, 2)
print("Question 2a:")
print("Data3 produces a {}% classification accuracy with kernelClassify".format(percent))


# In[14]:


# Question 2 PART B
# Testing on SuppVects
accuracy = testKernelClassify(suppVec_xVal, suppVects, alphas, b3, suppVec_labels)
percent = round(accuracy * 100, 2)
print("Question 2b:")
print("Support Vectors produces a {}% classification accuracy with kernelClassify".format(percent))


# In[15]:


# Question 3 
# Using 'data1'


# In[16]:


# gradient Descent helper function
# the derivative of the SVM optimization function: 
# minimizing the w[j] value of the hyperplane
def _gradientDescent(xI_j, wj, lamb, label):
    result = 2 * wj
    
    if label == 1:
        lamb *= -1
        
    result += lamb * xI_j
    
    return result


# In[17]:


# learning W hyperplane through gradient descent
# accepts data and number of iterations as arguments
def learnW(dataTrain, iters):
    # constant lambda and epsilon values
    lambConst = 0.1
    eps = 0.001
    
    # init an empty w of 0's
    w = np.zeros(7)
    
    # iterate arbitrary number of times (given by iters)
    for d in range(iters):
        
        # go through every data point
        for i in range(len(dataTrain)):
            dataPt = dataTrain[i]
            
            # store dataPt as features vector, with 1 appended to end
            x_i = dataPt[:-1]
            x_i = np.append(x_i, 1)
            
            # grab label as separate variable
            x_label = dataPt[-1]
            
            # gradient Descent towards the best w, updating over every feature
            for j in range(len(x_i)):
                w[j] -= eps * _gradientDescent(x_i[j], w[j], lambConst, x_label)
            
    return w


# In[18]:


# some test runs with varying iter
w_500 = learnW(data1, 500)
w_100 = learnW(data1, 100)
w_50 = learnW(data1, 50)
w_5 = learnW(data1, 5)
w_1 = learnW(data1, 1)


# In[19]:


# testing convergence
print(w_500 == w_100)


# In[20]:


# store values to test learnW()
w = w_500[:-1]
b = w_500[-1]


# In[46]:


# Q3 testing
# Testing learnW accuracy on data1
correct = 0

for i in data1:
    x = i[:-1]
    label = i[-1]
    
    result = _dotProduct(x,w) + b
    
    if result < 0 and label == -1:
        correct += 1
    elif result > 0 and label == 1:
        correct += 1

accuracy = round(correct/len(data1) * 100, 2)
print("learnW() accuracy on data1 after 500 iterations: {} %".format(accuracy))


# In[22]:


# Question 4
# using 'data1'


# In[23]:


# gradient Ascent helper function
# the derivative of the SVM optimization function with respect to lambda: 
# either [1 - (w^Tx + b)] or [1 + w^Tx + b]
def _gradientAscent(x, w, b, label):
    dot = _dotProduct(w,x) + b
    
    if label == 1:
        return 1 - dot
    else:
        return 1 + dot


# In[24]:


# now learning both w and best lambda for each datapoint

# wInit is the initial w Vector: shape (1,7)
# lamInit is the initial lambda values: shape (n,1)
# also accepts data and number of iterations
def learnWlam(dataTrain, wInit, lamInit, iters):
    # constant epsilon value, copy of lambda vector and w vector
    eps = 0.001
    lamVect = lamInit
    w = wInit
    
    # loop over dataset, iters number of times
    for d in range(iters):
        
        # go through every data point
        for i in range(len(dataTrain)):
            dataPt = dataTrain[i]
            
            # store dataPt as features vector, with 1 appended to end
            x_i = dataPt[:-1]
            x_i = np.append(x_i, 1)
            
            # grab label as separate variable
            x_label = dataPt[-1]

            # gradient descent on the best w value
            for j in range(len(x_i)):
                w[j] -= eps * _gradientDescent(x_i[j], w[j], lamVect[i], x_label)
            
            # gradient ascent on the best lambda value for respective dataPt
            lamVect[i] += eps * _gradientAscent(x_i[:-1], w[:-1], w[-1], x_label)
            
            # lambda must be non-negative
            if lamVect[i] < 0:
                lamVect[i] = 0 
            
    return {'wFin': w, 'lambdaFin': lamVect}


# In[26]:


# Question 5
# Testing learnWlam() after 5000 iterations


# In[30]:


# CAUTION: THIS CELL TAKES A LONG TIME

# init a zero vector for w
w_Q4 = np.zeros(7)

# init lambdas to 0.01
lambdas_Q4 = np.full((len(data1), 1), 0.01)

# get the dictionary for 5000 iterations
wLamDict_5000 = learnWlam(data1, w_Q4, lambdas_Q4, 5000)


# In[43]:


print("W vector for 5000 iterations: ", wLamDict_5000['wFin'][:-1])
print("B value for 5000 iterations: ", wLamDict_5000['wFin'][-1])


# In[47]:


correct = 0

w_testQ5 = wLamDict_5000['wFin']

for dataPt in data1:
    xi = dataPt[:-1]
    xi = np.append(xi, 1)
    
    yi = dataPt[-1]
    
    # w^Tx + b
    sign = _dotProduct(xi, w_testQ5)
    
    if sign > 0 and yi == 1:
        correct += 1
    elif sign < 0 and yi == -1:
        correct += 1
        
accuracy = round(correct/len(data1) * 100, 2)
print("Accuracy for learnWlam() with 5000 iters: {} %".format(accuracy))

