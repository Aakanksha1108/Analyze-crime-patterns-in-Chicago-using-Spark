#Spark HW - Question Number 2b

# Importing packages
from pyspark.sql import SQLContext
import csv
from pyspark import SparkContext
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.sql.types import TimestampType
from pyspark.sql.types import StringType
from datetime import datetime
import numpy as np
from datetime import date
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import pandas as pd
from pyspark.mllib.stat import Statistics

# Establishing connection
sc = SparkContext()

# A. Read dataset from path provided
path = 'hdfs://wolf.analytics.private/user/asp9295/Assignment_Spark/Crimes_-_2001_to_present.csv'
crime = sc.textFile(path)

# B. Filter data for 2015 to 2019
crime_v2 = crime.map(lambda row: row.split(","))
crime_v3 = crime_v2.filter(lambda row: row[17]=="2015" or row[17]=="2016" or row[17]=="2017" or row[17]=="2018" or row[17]=="2019")

# C. Keep relevant columns. For this exercise we only need beat and year. 
# They should together form the key as data needs to be rolled up at that level
crime_v4 = crime_v3.map(lambda row:((row[10],row[17]),1)).filter(lambda row: row[0][0].isdigit()==True)

# D. Rollup the data to beat-year level
crime_rollup = crime_v4.reduceByKey(lambda x,y:x+y)

# E. Cleaning data - Since some beat-year combinations may not have data, that will not allow us to compute correlations due to different sizes vectors
# Step 1: Take unique beats and years
unique_beats = crime_rollup.map(lambda row: row[0][0]).distinct().sortBy(lambda row: row)
unique_years = crime_rollup.map(lambda row: row[0][1]).distinct().sortBy(lambda row: row)

# Step 2: Calculate their cartesian product, make them key and assign a value of 0 to them
beats_cartesian_years = unique_beats.cartesian(unique_years).map(lambda row: (row,0))

# Step 3: Now left join the cartesian table with what we computed earlier to fill values for those records for which we already have data
master_data_v1 = beats_cartesian_years.leftOuterJoin(crime_rollup)

# Step 4: Series of manipulations to replace missing year-beat combinations values with 0 and retain the original counts for others
beat_level = master_data_v1.map(lambda row: (row[0],0) if row[1][1] is None else (row[0],row[1][1])).map(lambda row: (row[0][0], (row[0][1], row[1]))).groupByKey()

# The year values here are sorted to ensure that we have consecutive year values
beat_year_level = beat_level.map(lambda row:(row[0], sorted(row[1],key = lambda row:row[0]))).map(lambda row: (row[0],np.array([x[1] for x in row[1]])))

# F. Compute correlations
# Step 1: Ensure you have all possible combination of beats. Cartesian acts like a self join
beat_year_cross_rdd = beat_year_level.cartesian(beat_year_level)

# Step 2: Compute correlations and bring them in the format beat1, beat2, corr_value
beat1_beat2_corr = beat_year_cross_rdd.map(lambda row: (row[0][0],row[1][0],np.corrcoef(row[0][1], row[1][1]))).map(lambda row: (row[0], row[1], row[2][0][1]))

# Think of beat1_beat2_corr as a correlation matrix. Here, we remove all diagonal elements and lower triangle elements to ensure no repetition
beat1_beat2_upper_triangle = beat1_beat2_corr.filter(lambda row: row[0]!=row[1] and row[0]<row[1])

# Sort based on descending order of correlation value
corrs = beat1_beat2_upper_triangle.map(lambda row: ((row[0],row[1]),row[2])).sortBy(lambda row: -row[1]).collect()

# Filter top 400 correlations
top_corrs = corrs[:300]

with open('Sah_exercise2b.txt', 'w') as f:
    f.write('Beat1, \t Beat2, \t Correlation \n')
    writer = csv.writer(f)
    for line in top_corrs:
    	writer.writerow(line)

