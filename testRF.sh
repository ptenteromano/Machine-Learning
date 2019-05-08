#!/bin/bash
#../start.sh
source ../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /proj2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /proj2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/adult.data.csv /proj2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./bonus_randforest.py hdfs://$SPARK_MASTER:9000/proj2/input/
#../stop.sh
