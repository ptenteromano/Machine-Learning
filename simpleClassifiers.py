# Philip Tenteromano
# hw0
# 1/26/2018

def threshClassify(heightList, xThresh):
  giraffes = []
  for animal in heightList:
    if animal > xThresh:
      giraffes.append(1)
    else:
      giraffes.append(0)
  return giraffes

# test 1
heights = [4, 3, 2, 8, 1, 12]
longNecks = threshClassify(heights, 3)
print(longNecks)

def findAccuracy(classifierOutput, trueLabels):
  matches = [1 for i, j in zip(classifierOutput, trueLabels) if i == j]
  return len(matches) / len(classifierOutput)
  
# test 2
correct = [1, 0, 1, 1, 1, 0] # used for trueLabels
correctness = findAccuracy(longNecks, correct)
print(correctness)

# passing in 2d list (2 rows by C columns)
def getTraining(fullData):
  training = [[], []]
  for r in range(len(fullData)):
    for c in range(len(fullData[r]) // 3):
      training[r].append(fullData[r][c])
  return training

# test 3
data = [heights, correct]
trainingData = getTraining(data)
print(trainingData)
