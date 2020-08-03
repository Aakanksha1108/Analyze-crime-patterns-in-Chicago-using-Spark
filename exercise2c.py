# Assignment Spark : Q2c

# Importing packages

from datetime import datetime
import csv
from datetime import date
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext()
sqlcontext = SQLContext(sc)

# Define the path
path = 'hdfs://wolf.analytics.private/user/asp9295/Assignment_Spark/Crimes_-_2001_to_present.csv'

# Read the file
crime = sc.textFile(path)

# Split the line items
header = crime.first() # Extract Header
crime_v2 = crime.filter(lambda row: row != header)
crime_v3 = crime_v2.map(lambda line: line.split(","))

# Subset the RDDs based on dates to obtain data relevant for mayor emanuel and daley
crime_v4 = crime_v3.map(lambda x: (x[12],datetime.strptime(x[2],'%m/%d/%Y %I:%M:%S %p').date()))
mayor_daley_rdd = crime_v4.filter(lambda x: x[1] <= date(2011, 5, 16)).persist()
mayor_emanuel_rdd = crime_v4.filter(lambda x:(x[1]>date(2011,5,16))).persist()

# Rollup the datasets at ward-month level
# The reason for analyzing this at month level is that we have seen crime has a seasonal pattern and in order to ensure a similar comparison,
# it would be a nice idea to look at the number of crimes that happened under their supervision at this level.
# We cannot look at a yearly level as they ruled during different years
# The choice of ward level for this analysis was because it is granular enough and we have just done a lot ot beat and block level already. So, just for a different flavor

temp1 = mayor_emanuel_rdd.map(lambda x: ((x[0], x[1].month), 1))
emanuel_summary = temp1.reduceByKey(lambda x,y: x+y)
temp2 = mayor_daley_rdd.map(lambda x: ((x[0], x[1].month), 1))
daley_summary = temp2.reduceByKey(lambda x,y: x+y)

# Normalizing dataset as the two mayors correspond to different time durations. This is very important since we have more data for Major Daley and just comparing the sums would give wrong information about him
e_yrs = mayor_emanuel_rdd.map(lambda x: x[1].year).distinct().count()
d_yrs = mayor_daley_rdd.map(lambda x: x[1].year).distinct().count()
e_final = emanuel_summary.map(lambda x:(x[1]/e_yrs))
d_final = daley_summary.map(lambda x:(x[1]/d_yrs))
daley = d_final.collect()
emanuel = e_final.collect()

# Performing a t-test
diff = [x - y for x,y in zip(daley, emanuel)]
mean = sum(diff)/len(diff)
t2, p2 = stats.ttest_ind(daley,emanuel)

with open('Sah_exercise2c.txt', 'w') as f:
	f.write("Difference in the ward-month level crimes between Mayor Daley and Mayor Emanuel\n")
	f.write(str(mean))
	f.write("\n")
	f.write("T-test value to assess significance of this difference:\n")
	f.write(str(t2))
	f.write("\n")
	f.write("P-value of t-test:\n")
	f.write(str(p2))
