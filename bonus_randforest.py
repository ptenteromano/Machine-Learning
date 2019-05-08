# Philip Tenteromano
# Antonio Segalini
# Wenjia Zheng
# Yun Song

# 5/5/2019
# Big Data Programming

# Project 2 - Bonus
# Random Forest

from sys import argv
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, OneHotEncoderModel
from pyspark.ml import Pipeline

# set up the session 
spark = SparkSession\
    .builder\
    .appName("KMeansExample")\
    .getOrCreate()

# Taken from .names description file
col_names = ['Age','Workclass','FinalWeight','Education','EducationNum','MaritalStatus','Occupation','Relationship','Race',
         'Sex','CapitalGain','CapitalLoss','HoursPerWeek','NativeCountry','income']

# store file into a pyspark dataframe
df = spark.read.format("csv").load(argv[1], header='false', names=col_names, inferSchema="true")
df = df.toDF(*col_names)

# EducationNum works better than categorical 'Education'
df = df.drop('Education')
cols = df.columns

catColumns = [ item[0] for item in df.dtypes if item[1].startswith('string') ]
catColumns = [ c for c in catColumns if 'income' not in c ]

numColumns = [ item[0] for item in df.dtypes if not item[1].startswith('string') ]
catColVectors = [c + '_vector' for c in catColumns ]

# Change categorical values into numeric
indexers = [ StringIndexer(inputCol=column, outputCol=column+"_index") for column in catColumns ]
encoder = OneHotEncoderEstimator(
    inputCols=[c + "_index" for c in catColumns],
    outputCols=[c + "_vector" for c in catColumns]
    )

assembler = VectorAssembler(
    inputCols=encoder.getOutputCols() + numColumns,
    outputCol="features"
    )

label_stringIdx = StringIndexer(inputCol="income", outputCol="label")

pipeline = Pipeline(stages=indexers + [label_stringIdx, encoder, assembler])
encoded_df = pipeline.fit(df).transform(df)

selectedCols = ['label', 'features'] + cols
dataset = encoded_df.select(selectedCols)

# Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())


# Fit model and train
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
rf2 = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

model = rf.fit(trainingData)
model2 = rf2.fit(trainingData)
predictions = model.transform(testData)
predictions2 = model2.transform(testData)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
accuracy2 = evaluator.evaluate(predictions2)

print("\n\tShowing Random Forest Results: \n")
print("\tAccuracy with 10 trees: %s\n\t" % (accuracy))
print("\tAccuracy with 100 trees: %s\n\t" % (accuracy2))



print('\nDone')
