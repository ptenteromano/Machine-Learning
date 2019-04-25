# These functions return the probablity of 'value' in the given distribution

def gaussian(value, μ, 𝜎):
    # Calculate variance
    𝜎2 = 𝜎 * 𝜎

    # Calculate exponentiated term
    exp = np.power(np.e, -(np.power(value - μ, 2) / (2 * 𝜎2)))

    # Calculate denominator term
    den = np.power((np.sqrt(2 * np.pi * 𝜎2)), -1)

    # Calculate probability from both values above
    prob = den * exp

    return prob

# 'a' --> lower bound on the interval
# 'b' --> upper bound on the interval
def uniform(value, a, b):
    return 1/(b - a) if (a <= value <= b) else 0

def rayleigh(value, 𝜎):
    
    # Rayleigh only defined for positive values
    if value < 0:
        return 0
    
    # Left side part
    multiplicand_a = value/(𝜎**2)
    
    # Exponent on the exponential
    exponent = (-value**2)/(2 * 𝜎**2)
    
    # Right side part
    multiplicand_b = np.exp(exponent)
    
    # Return Rayleigh value
    return multiplicand_a * multiplicand_b



# Classify using MLE

# Parameter Notes:
# featuresData --> the dataset without label column
# means --> a length 2 array with feature means for [ wins, losses ]
# stdDev --> a length 2 array with feature means for [ wins, losses ]
# 'a' --> array of lower bounds on each feature's interval
# 'b' --> array of upper bound on each feature's interval

# MLE with Guassian PDF
def gaussian_classify(featureData, means, stdDev):
    
    # Create an empty column of predictedLabels
    n = featureData.shape[0]
    predictedLabels = np.zeros((n,))
    
    # Iterate over entire dataset
    for loc, featureList in enumerate(featureData.values):
        probOfWin = 1
        probOfLoss = 1
        
        # Calculate probability of values in win / loss distributions
        for i, feature in enumerate(featureList):
            probOfWin *= gaussian(feature, means[0][i], stdDev[0][i])
            probOfLoss *= gaussian(feature, means[1][i], stdDev[1][i])
        
        # Choose highest probability
        predictedLabels[loc] = 1 if probOfWin > probOfLoss else -1

    return predictedLabels
            
          
# MLE with uniform PDF
def uniform_classify(featureData, a, b):

    # Create an empty column of predictedLabels
    n = featureData.shape[0]
    predictedLabels = np.zeros((n,))
    
    # Iterate over entire dataset
    for loc, featureList in enumerate(featureData.values):
        probOfWin = 1
        probOfLoss = 1
        
        # Calculate probability of values in win / loss distributions
        for i, feature in enumerate(featureList):
            probOfWin *= uniform(feature, a[0][i], b[0][i])
            probOfLoss *= uniform(feature, a[1][i], b[1][i])

        # Choose highest probability
        predictedLabels[loc] = 1 if probOfWin > probOfLoss else -1

    return predictedLabels
    
    
# MLE with rayleigh PDF
def rayleigh_classify(featuresData, stdDev):
    
    # Epsilon for zero-values
    ε = 0.00001
    
    
    # Create an empty column of predictedLabels
    predictedLabels = __fill(len(featuresData), 0)
    
    # Iterate over entire dataset
    for loc, featuresList in enumerate(featuresData.values):
        
        # Instantiate initial probabilities
        probOfWin = 1
        probOfLoss = 1
        
        
        # Calculate probability of values in win / loss distributions
        for i, feature in enumerate(featuresList):
            
            # Pad the feature value with epsilon if it's equal to 0 to avoid
            # knocking out the probability value.
            if feature == 0:
                feature = ε
            
            probOfWin = rayleigh(feature, stdDev[0][i])            
            probOfLoss = rayleigh(feature, stdDev[1][i]) 
            
        # Choose highest probability
        predictedLabels[loc] = 1 if probOfWin >= probOfLoss else -1
        
    return predictedLabels