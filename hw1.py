
# coding: utf-8

# In[1]:


# Philip Tenteromano
# Machine Learning
# CISC 5800
# 2/9/2018
# hw1.py

# using jupyter notebook
# For the following, Standard Deviation is given to be 2


# In[2]:


import numpy as np
import scipy.io as sio


# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


# load our data into numpy arrays
data = sio.loadmat('hw1data.mat')
training = np.array(data['trainData'])
testing = np.array(data['testData'])


# In[5]:


# 1st column represents shopper class - 1 for Minor, 2 for Youth, 3 for Adult, 4 for Senior
# 2nd column represents corresponding amount of alcohol
testing.shape
training.shape


# In[6]:


# checking on data
training


# In[7]:


# for plotting purposes, slice all classes by themselves
minors = training[training[:,0] == 1]
youth = training[training[:,0] == 2]
adults = training[training[:,0] == 3]
seniors = training[training[:,0] == 4]


# In[8]:


# DISTRIBUTION COMMENT
# These histograms of the data show that these are normal distributions
# Their is a bell curve in each of them
# plt.hist(testing)
# plt.show()


# In[9]:


# plt.hist(minors)
# plt.hist(youth)
# plt.hist(adults)
# plt.hist(seniors)
# plt.show()


# In[10]:


# all classes
# plt.hist(training)
# plt.show()


# In[11]:


# get a mean from training data, provided a class value (1-4)
# added soda for Question 8
def learnMean(Data, classNum, soda=None):
    # use slicing
    classType = Data[Data[:,0] == classNum]
    
    if soda:
        return classType[:,2].mean()
    else:
        return classType[:,1].mean()


# In[12]:


# testing on adults class
learnMean(training, 3) == training[training[:,0] == 3][:,1].mean()


# In[13]:


# create meanVector [1, 2, 3, 4]
means = np.array([learnMean(training, 1), learnMean(training, 2), learnMean(training, 3), learnMean(training, 4)])
means


# In[14]:


# another version of the gaussian PDF without using variance
# def otherGaussPDF(x, mean, std):
#     return (1/(std*np.sqrt(2*np.pi))) * (np.e ** (-.5 * ((x - mean) / std) ** 2))


# In[15]:


# the gaussian PDF helper function - returns probability in range (0, 1) for normal distributions
# three parameters required - the proposed value, with the mean and std of the curve
def gaussianPDF(x, mean, std):
    vari = std ** 2
    return (1/(np.sqrt(2*np.pi*vari))) * (np.e ** (-((x - mean) ** 2) / (2 * vari)))


# In[16]:


# Max Likelihood Estimation - MLE
def labelML(amountAlc, meanVector):
    # char symbol for respective classValues
    classes = ['M','Y','A','S']
    maximum = 0
    index = 0
    
    # find the max probability against all distributions
    for idx, mean in enumerate(meanVector):
        prob = gaussianPDF(amountAlc, mean, 2)
        
        if prob > maximum:
            maximum = prob
            index = idx
    
    # return the char associated with max
    return classes[index]
        


# In[17]:


# testing MLE
labelML(9, means)


# In[18]:


# create prior vector for MAP
prior = [0.3, 0.4, 0.2, 0.1]


# In[19]:


# Max a Posteriori - MAP - multiply PDF by prior (prior effects our probability)
def labelMP(amountAlc, meanVector, priorVector):
    # char symbol for respective classValues
    classes = ['M','Y','A','S']
    maximum = 0
    index = 0
    
    # with prior influence, find the max probability against all distributions
    for idx, mean in enumerate(meanVector):
        prob = gaussianPDF(amountAlc, mean, 2) * priorVector[idx]
        
        if prob > maximum:
            maximum = prob
            index = idx
    
    # return the char associated with max
    return classes[index]


# In[20]:


# testing MAP
labelMP(10, means, prior)


# In[21]:


# MLE - returns fraction of correctly labeled testData points against training data
def evaluateML(testData, meanVector):
    # assign key-value for classes return value
    classes = {'M': 1, 'Y': 2, 'A': 3, 'S': 4}
    correct = 0
    
    # check if testData's label is correct against MLE on the mean Training vector
    for (testClass, alc) in testData:
        label = labelML(alc, meanVector)
        
        if classes[label] == testClass:
            correct += 1
            
    return correct / testData.shape[0]


# In[22]:


# MLE - using testData against entire training MeanVector 
evaluateML(testing, means)


# In[23]:


# MAP - returns fraction of correctly labeled testData points against training
def evaluateMP(testData, meanVector, priorVector):
    classes = {'M': 1, 'Y': 2, 'A': 3, 'S': 4}
    correct = 0

    # check if testData's label is correct against MAP on the mean Training vector
    for (testClass, alc) in testData:
        label = labelMP(alc, meanVector, priorVector)
        
        if classes[label] == testClass:
            correct += 1

    return correct / testData.shape[0]


# In[47]:


# MAP - using testData against entire training MeanVector
evaluateMP(testing, means, prior)


# In[44]:


# reporting for MLE with means of the first 6, 18, 54, 162 of training samples
first_n = 6

print("Testing MLE on first-n slices:")
for i in range(4):
    means_n = np.array([learnMean(training[:first_n], 1), learnMean(training[:first_n], 2), learnMean(training[:first_n], 3), learnMean(training[:first_n], 4)])
    print("First {}: ".format(first_n), evaluateML(testing, means_n))

    # increment slice by multiple of 3 for next test
    first_n *= 3
print('\n')


# In[49]:


# reporting for MAP with means of first 6, 18, 54, 162
first_n = 6

print("Testing MAP on first-n slices:")
for i in range(4):
    means_n = np.array([learnMean(training[:first_n], 1), learnMean(training[:first_n], 2), learnMean(training[:first_n], 3), learnMean(training[:first_n], 4)])
    print("First {}: ".format(first_n), evaluateMP(testing, means_n, prior))
    
    # increment slice by multiple of 3 for next test
    first_n *= 3

print('\n')
# In[27]:


# Question 7 REPORT
# Above you can see both MLE and MAP for the means of the first 6, 18, 54, and 162 samples of training data
# 
# MLE:
#     The MLE seems to be getting worse as the samples increase. It started with 0.54, and continued to decrease.
#
# MAP:
#     The MAP seemed low at first (because prior has a strong effect with little data), but was able to become 
#     it's most accurate with the 54 data samples


# In[28]:


# Question 8
data = sio.loadmat('hw1dataQ8.mat')
testing_2 = data['testData']
training_2 = data['trainData']


# In[29]:


# col1 = classVal (1-4), # col2 = amtAlc, # col3 = amtSoda
training_2


# In[30]:


# amount drinks [alc, soda]
training_2[:,1:]


# In[31]:


# meansMatrix shape (2,4)
# first row is alc mean, second row is soda mean
drinkMeanMatrix = np.array([[learnMean(training_2, 1), learnMean(training_2, 2), learnMean(training_2, 3), learnMean(training_2, 4)],
                            [learnMean(training_2, 1, True), learnMean(training_2, 2, True), learnMean(training_2, 3, True), learnMean(training_2, 4, True)]])

drinkMeanMatrix


# In[32]:


# amount drinks is array with [amtAlc, amtSoda]
def labelMP2(amountDrinks, meansMatrix, priorVector):
    classes = ['M','Y','A','S']
    maximum = 0
    index = 0
    
    for i in range(meansMatrix.shape[1]):
        # find a probability on both the alcohol axis and the soda axis
        alcProb = gaussianPDF(amountDrinks[0], meansMatrix[0, i], 2)
        sodaProb = gaussianPDF(amountDrinks[1], meansMatrix[1, i], 2)
 
        # combine these probabilities with prior 
        prob = alcProb * sodaProb * priorVector[i]

        # check for max, return that label
        if prob > maximum:
            maximum = prob
            index = i
    
    return classes[index]


# In[51]:


labelMP2(testing_2[1,1:], drinkMeanMatrix, prior)


# In[ ]:


# running above function for the first few testData values 
classes = ['M', 'Y', 'A', 'S']
for i in range(5):
    label = labelMP2(testing_2[i,1:], drinkMeanMatrix, prior)
    real = int(testing_2[i,0])
    print("Predicted: ", label)
    print("Actual from test: ", real)
    
    if classes[real - 1] == label:
        print("Correct!")
    else:
        print("Incorrect.")
    print('\n')

