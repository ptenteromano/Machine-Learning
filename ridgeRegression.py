
# coding: utf-8

# In[1]:


# Philip Tenteromano
# 2/19/2019
# CISC 6930
# Data Mining
# Dr. Yijun Zhao
#
# Homework 2

# Written answers found below


# In[2]:


# start with imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# if the below doesn't work, you need to run HW2_DataSplit.py 

import HW2_DataSplit as files
files.produceDataFiles()

# for jupyter notebook
# get_ipython().run_line_magic('run', 'HW2_DataSplit.py')


# In[4]:


# import data into iterable form
# make arrays of our dataframes
trainingSets = []
testingSets = []

# 1-3 are full sets, 4-6 are the partitioned sets
trainFiles = ['train-100-10.csv', 'train-100-100.csv', 
              'train-1000-100.csv','train-50(1000)-100.csv', 
              'train-100(1000)-100.csv', 'train-150(1000)-100.csv']

testFiles = ['test-100-10.csv', 'test-100-100.csv', 
              'test-1000-100.csv']

# load data sets into array
for i in range(6):
    trainingSets.append(pd.read_csv(trainFiles[i]))
    
    # insert a column of '1's to beginning of df
    # for y-intercept calculation
    length = trainingSets[i].shape[0]
    y_ints = np.ones(length)
    trainingSets[i].insert(loc=0, column='yIntercepts', value=y_ints)
    
    
# we only have 3 test sets, index [2] will be used for the training partitions
for i in range(3):
    testingSets.append(pd.read_csv(testFiles[i]))
    
    # insert a column of '1's to beginning of df
    # for y-intercept calculation
    length = testingSets[i].shape[0]
    y_ints = np.ones(length)
    testingSets[i].insert(loc=0, column='yIntercepts', value=y_ints)


# In[5]:


# primary functions:


# In[6]:


# computation of the closed form: w = [ (X'X + lamb*I)**-1 (X'y) ]
# function will write it as: w = [ B**-1 (c) ] or [Bw = c]
def ridgeRegression(X, y, lambdaRange):
    wList = []
    rss = []
    
    # get normal form of `X`
    A = X.T @ X 
    
    # get Identity matrix
    I = np.eye(A.shape[0])
    
    # added this for part 4 - can take either list or integer range 
    if isinstance(lambdaRange, int):
        lambIterer = np.arange(1, lambdaRange+1)
    else:
        lambIterer = lambdaRange
    
    for lambVal in lambIterer:
        # set up equation Bw = c (remove inversion)     
        lamb_I = lambVal * I
        B = A + lamb_I
        c = X.T @ y
        
        # solve has shown to be more stable for computing [(B)**-1 (c)]
        # other option was np.linalg.inv(B) @ c
        w = np.linalg.solve(B,c)
        wList.append(w)
        
        # use w to find Mean of rss
        r = rssMean(X, y, w)
        
        rss.append(r)
        
    return (wList, rss)


# In[7]:


# use w to find residual sum of squares for each lambda value
# residual vector = [ (y - Xw).T @ (y - Xw) ]
# finally, get the mean of RSS (r / n)
def rssMean(X_mat, y_vect, w_vect):
    R = y_vect - (X_mat @ w_vect)
    r = R.T @ R
    r /= X_mat.shape[0]
    
    return r


# In[8]:


# runs the testSet against the trained w-vectors
# returns the MSE against each w-vector evaluated at lambda
def testRidge(x_, y_, w_Matrix, lambdaRange):
    testRSS = []
    
    if isinstance(lambdaRange, int):
        lambIterer = np.arange(0, lambdaRange)
    else:
        lambIterer = np.arange(len(lambdaRange))
    
    # we are indexing now (0-149)
    for i in lambIterer:
        r = rssMean(x_, y_, w_Matrix[i])
        testRSS.append(r)
        
    return testRSS


# In[9]:


# TRAINING (get w-vector and MSE)
# make a corresponding array mapping for matrix of lambda/coefficientsVector
# for every training set
w_Sets = []
mse_Sets_Train = []
_lamb = 150

# separate parameters (x) and y values 
for i in range(6):
    xTrain = trainingSets[i].iloc[:,:-1]
    yTrain = trainingSets[i].iloc[:,-1]
    
    # run ridge on Training data, store w-vector as coEff and rss as MSE mapped to lambda value
    coEff, mse = ridgeRegression(xTrain, yTrain, _lamb)
    
    # i.e. w_Sets[0] lines up with trainingSet[0]
    w_Sets.append(coEff)
    mse_Sets_Train.append(mse)


# In[10]:


# TESTING
mse_Sets_Test = []

# similiar concept for testing data
for i in range(3):
    x_Test = testingSets[i].iloc[:,:-1]
    y_Test = testingSets[i].iloc[:,-1]
    w_data = w_Sets[i]
    
    # we already have our w-vectors, run the testing set against the w-vectors
    m = testRidge(x_Test, y_Test, w_data, _lamb)
    
    mse_Sets_Test.append(m)

# TESTING the last test final against the 3 partitioned trainings
for i in range(3,6):
    x_Test = testingSets[2].iloc[:,:-1]
    y_Test = testingSets[2].iloc[:,-1]
    w_data = w_Sets[i]
    
    m = testRidge(x_Test, y_Test, w_data, _lamb)
    
    mse_Sets_Test.append(m)

# In[12]:


# can be used for plotting both the pairs and the partition datasets
def plotMSE(train_MSE, test_MSE, lambdaValues, partition=False):
    fig, axes = plt.subplots(figsize=(14,4))
    lambdaValues = range(1,lambdaValues+1)
    nrows, ncols = 1, 3
    plots_total = ncols
        
    for x in range(plots_total):
        i = x
        y = x
        if partition:
            y = 2
            i += 3
        ax = plt.subplot(nrows, ncols, x+1)
        ax.plot(lambdaValues, train_MSE[i], label='Train', color='Red')
        ax.plot(lambdaValues, test_MSE[y], label='Test', color='Blue', ls='-.')
        ax.legend()
        plt.title(testFiles[y][:-4] + ' vs ' + trainFiles[i][:-4])
        plt.xlabel('f ($\lambda$)')
        plt.ylabel('MSE')
        plt.grid()
    plt.tight_layout()
    plt.show()


# In[13]:


# plotting the first 3 train vs 3 test sets
plotMSE(mse_Sets_Train, mse_Sets_Test, _lamb)


# In[14]:


# plotting the test sets against the partitioned training sets
plotMSE(mse_Sets_Train, mse_Sets_Test, _lamb, partition=True)


# In[15]:


# returns the lambda value that minimizes testing MSE 
def findLambdaMinMSE(test_MSE):
    return str(np.argmin(test_MSE) + 1)


# In[16]:


# minimum MSE values for 3 test against original 3 train
print("Question #2 part a:")
for i,v in enumerate(testFiles):
    print("[{}] minimized MSE at lambda = {}".format(v[:-4], findLambdaMinMSE(mse_Sets_Test[i])))

print('\n')

# now against partitions
for i in range(3,6):
    print("[{}] minimized MSE at lambda = {} on set [{}]".format(testFiles[2][:-4], findLambdaMinMSE(mse_Sets_Test[i]), trainFiles[i][:-4]))


# In[17]:


# part b
lambdaValues = range(1,_lamb+1)

fig = plt.figure()

plt.plot(lambdaValues, mse_Sets_Test[1], label='100-100', color='blue', ls='--')
plt.plot(lambdaValues, mse_Sets_Train[3], label='50(1000)-100', color='green')
plt.plot(lambdaValues, mse_Sets_Train[4], label='100(1000)-100', color='purple', ls=':')

plt.legend()
plt.title('Q2: PartB')
plt.show()


# In[18]:


# QUESTION 2 ANSWERS
#
# PART C: 
# We can see that the testing set of 100-100 has very poor MSE as it is close to 0, this is due to the fact that
# the number of data points is actually equal to the number of parameters. This makes it very hard to fit a linear
# regression line (no lambda penalty) to a line with very few data entries. The penalty helps this.
# 
# The other lines are from training sets and their MSE increases very rapidly as lambda goes up because of the lack of 
# data entries to truly train the data. A perfect linear fit is easy with few data points, but can be very bad on test data.


# In[19]:


# start 3


# In[20]:


# 1. split data into 10 disjoint folds
# 2. for all lambda values (1-150)
# 3. Train on everything EXCEPT the ith fold
# 4. Test on the ith fold and record the error on fold i
# 5. COMPUTE AVG performance of lambda on 10 folds
# 6. Pick the value of lambda with the best avg performance


# In[21]:


foldError = []
trainCV_Lambda = []
trainCV_MSE = []

def crossValRidge(dataSet, numFolds):
    dataSegment = dataSet.shape[0] // numFolds
    lastRow = dataSet.shape[0]

    mseSums = []

    for i in range(numFolds):
        lowerSlice = int(i * dataSegment)
        upperSlice = int(lowerSlice + dataSegment)

        # filter out the CV test data
        ith_test = dataSet.iloc[lowerSlice:upperSlice]

        # split training around the CV testdata
        trainA = dataSet.iloc[0:lowerSlice]
        trainB = dataSet.iloc[upperSlice:lastRow]

        # re-combine the training data
        newTrain = pd.concat([trainA,trainB])

        # slice out x,y from train and test data
        x = newTrain.iloc[:,:-1]
        y = newTrain.iloc[:,-1]

        xTest = ith_test.iloc[:,:-1]
        yTest = ith_test.iloc[:,-1]

        # run ride on training
        w_CV, mse_train = ridgeRegression(x, y, _lamb)

        # get the MSE for the ith-train data
        mse_CV_test = testRidge(xTest, yTest, w_CV, _lamb)

        # record error, append to index of fold
        foldError.append(mse_CV_test)

        # fill and sum MSE
        if not mseSums:
            mseSums += mse_CV_test
        else:
            mseSums = [sum(x) for x in zip(mseSums, mse_CV_test)]

    # take average of MSE
    mseSums = [x/numFolds for x in mseSums]

    # lambda value
    trainCV_Lambda.append(np.argmin(mseSums) + 1)

    # actual MSE value
    trainCV_MSE.append(min(mseSums))


# In[22]:


foldError = []
trainCV_Lambda = []
trainCV_MSE = []

# run crossValRidge on every training set
for i in trainingSets:
    crossValRidge(i, 10)


# In[23]:


# QUESTION 3

print("Question 3 part A:\n")

for i,v in enumerate(trainCV_Lambda):
    print("[{}] minimizes MSE at lambda = {}. MSE value minimized at {}".format(trainFiles[i][:-4], v, trainCV_MSE[i]))


# In[24]:


# QUESTION 3 part b
# 
# We can say that the cross validation technique kept the MSE relatively very low (they are all less than 7.5, and as low as 5.1)
# Also, the lambda values stayed relatively close until the data points grew large (1000 samples)
# Without cross validation, the lambda values seemed to stray very far from each other
# That can be bad for real-world applications! - Use CV!
# 
# part c
# CV has drawbacks if the training set and validation set are drawn from a similiar population
# If the data varies a lot, then it can yield some unmeaningful results
# It also takes an algorithmically complex time to constantly slice and evaluate data

# part d
# The factors effecting performance for CV rely heavily on how many folds you make when using k-fold CV
# Becuase you have to re-slice and train your data everytime, when datasets become large, and/ or k becomes large,
# it can cause the algorithmn to become slow. An increase in k means that many more times the training must be run
# and the testing must be tested


# In[25]:


import random


# In[26]:


# Question 4

# get our variables, using '1000-100.csv' sets
lambs = [1, 25, 150]
trainings = trainingSets[2]
testings = testingSets[2]

sizeSlices = [30,120,210,290,380,470,560,650,740,830,920,1000]

def learningCurve(train, test, lambs, sizeSlices):
    upperBound = train.shape[0]
    mseData = []
    
    for i in sizeSlices:
        # create a random int to slice from
        if i == upperBound:
            rand = 0
        else:
            rand = random.randint(1, upperBound - i)
        
        # slice out a random of portion with the given size
        upperSlice = i + rand
        trainSlice = train.iloc[rand:upperSlice]
        
        # split x and y
        x = trainSlice.iloc[:,:-1]
        y = trainSlice.iloc[:,-1]
        
        w_Vect, mse_train = ridgeRegression(x, y, lambs)
        
        xTest = test.iloc[:,:-1]
        yTest = test.iloc[:,-1]
        mse_test = testRidge(xTest, yTest, w_Vect, lambs)
        
        mseData.append(mse_test)

    return mseData


# In[27]:


plotq4mse = learningCurve(trainings, testings, lambs, sizeSlices)
plotq4mse = np.asarray(plotq4mse).transpose()


# In[28]:


# Question 4 plot
# part b

fig = plt.figure()

plt.plot(sizeSlices, plotq4mse[0], label='Lambda 1', color='blue')
plt.plot(sizeSlices, plotq4mse[1], label='Lambda 25', color='green', ls='--')
plt.plot(sizeSlices, plotq4mse[2], label='Lambda 150', color='purple', ls=':')

plt.ylabel('MSE')
plt.xlabel('Training Sample Size')
plt.ylim(4,14)

plt.legend()
plt.title('Q4: Random sampling')
plt.show()

# In[29]:


# done

