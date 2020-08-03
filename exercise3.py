# Importing packages

import csv
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType, TimestampType, StringType
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd
from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext()
sqlcontext = SQLContext(sc)

# A. Read dataset from path provided
path = 'hdfs://wolf.analytics.private/user/asp9295/Assignment_Spark/Crimes_-_2001_to_present.csv'
crime = sqlcontext.read.csv(path, header = True)

# B. Extracting temporal information from the records

# Step 1: Convert the Date time column to a usable format
func =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'), DateType())
crime = crime.withColumn('Date', func(col('Date')))

# Step 2: Extract week of year information from the date column
crime = crime.withColumn('week_of_year',weekofyear(crime['Date']))

# Step 3: Convert year to integer. The below command will overwrite the existing year column which is fine as we need to convert that to integer anyways
crime = crime.withColumn('year',year(crime['Date']))

# C. Add a flag to denote if the crime is violent or not

violent_iucr_codes = ["0110","0130","0261","0262","0263","0264","0265","0266","0271","0272","0273","0274","0275","0281","0291","1753","1754","0312",'0313','031A','031B','0320','0325','0326',
'0330','0331','0334','0337','033A','033B','0340']
crime = crime.withColumn("violent_flag",when(crime["IUCR"].isin(violent_iucr_codes),lit(1)).otherwise(lit(0)))

# D. Separate the data into 2 datasets (one consisting of violent crimes only and the other consisting of all other crimes)

crime = crime.withColumn("temp", lit(1))
crime_nv = crime.filter(col("violent_flag")==0)
crime_v = crime.filter(col("violent_flag")==1)

# E. Roll up the datasets to the level at which we will make predictions

crime_v_agg = crime_v.groupBy("beat","year","week_of_year").agg(count("*").alias("num_crimes"))
crime_nv_agg = crime_nv.groupBy("beat","year", "week_of_year").agg(count("*").alias("num_crimes"))

crime_v_agg = crime_v_agg.withColumn("year",crime_v_agg['year'].cast(StringType()))
crime_v_agg = crime_v_agg.withColumn("week_of_year",crime_v_agg['week_of_year'].cast(StringType()))
crime_nv_agg = crime_nv_agg.withColumn("year",crime_nv_agg['year'].cast(StringType()))
crime_nv_agg = crime_nv_agg.withColumn("week_of_year",crime_nv_agg['week_of_year'].cast(StringType()))

# F. Create lag features using the column "num_crimes"

# Step 1: Sort data so that the lag variables convey information about the same level
crime_v_agg_sorted = crime_v_agg.orderBy("year","week_of_year")
crime_nv_agg_sorted = crime_nv_agg.orderBy("year","week_of_year")

# Step 2: Create lag variables
window = Window().partitionBy().orderBy(['year', 'week_of_year'])
crime_v_f = crime_v_agg_sorted.withColumn('num_crimes_lag_1', lag(col('num_crimes'), count=1).over(window)).na.drop()
crime_v_f = crime_v_f.withColumn('num_crimes_lag_2', lag(col('num_crimes'), count=2).over(window)).na.drop()
crime_v_f = crime_v_f.withColumn('num_crimes_lag_3', lag(col('num_crimes'), count=3).over(window)).na.drop()
crime_v_f = crime_v_f.withColumn('num_crimes_lag_4', lag(col('num_crimes'), count=4).over(window)).na.drop()
crime_v_f = crime_v_f.withColumn('num_crimes_lag_5', lag(col('num_crimes'), count=5).over(window)).na.drop()

crime_nv_f = crime_nv_agg_sorted.withColumn('num_crimes_lag_1', lag(col('num_crimes'), count=1).over(window)).na.drop()
crime_nv_f = crime_nv_f.withColumn('num_crimes_lag_2', lag(col('num_crimes'), count=2).over(window)).na.drop()
crime_nv_f = crime_nv_f.withColumn('num_crimes_lag_3', lag(col('num_crimes'), count=3).over(window)).na.drop()
crime_nv_f = crime_nv_f.withColumn('num_crimes_lag_4', lag(col('num_crimes'), count=4).over(window)).na.drop()
crime_nv_f = crime_nv_f.withColumn('num_crimes_lag_5', lag(col('num_crimes'), count=5).over(window)).na.drop()

# G. Treat "Beat" as a categorical variable

# Indexer
BeatIdxer = StringIndexer(inputCol="beat", outputCol="BeatIdx").setHandleInvalid("keep")
# One-hot encoder
encoder = OneHotEncoderEstimator(inputCols=["BeatIdx"], outputCols=["BeatVec"]).setHandleInvalid("keep")

# H. Build ML pipeline

# Vector assembler
assembler = VectorAssembler(
    inputCols=["BeatVec","num_crimes_lag_1", "num_crimes_lag_2", "num_crimes_lag_3", "num_crimes_lag_4","num_crimes_lag_5"],
    outputCol="features",)

# Random Forest model
rf = RandomForestRegressor(labelCol="num_crimes", featuresCol="features")

# Defining Pipeline
pipeline = Pipeline(stages=[BeatIdxer, encoder, assembler, rf])

# I. Fit Model

# Split train test
train_v, test_v = crime_v_f.randomSplit([0.75, 0.25])
train_nv, test_nv = crime_nv_f.randomSplit([0.75, 0.25])

# Train model for violent crimes prediction
rf_v = pipeline.fit(train_v)
pred_v = rf_v.transform(test_v)

# Train model for non-violent crimes prediction
rf_nv = pipeline.fit(train_nv)
pred_nv = rf_nv.transform(test_nv)

# J. Measure accuracy

# Calculate rmse for violent crimes
evaluator_v = RegressionEvaluator(labelCol="num_crimes", predictionCol="prediction", metricName="rmse",)
rmse_v = evaluator_v.evaluate(pred_v)

# Calculate r2 for violent crimes
evaluator2_v = RegressionEvaluator(labelCol="num_crimes", predictionCol="prediction", metricName="r2",)
r2_v = evaluator2_v.evaluate(pred_v)

# Calculate rmse for non-violent crimes
evaluator_nv = RegressionEvaluator(labelCol="num_crimes", predictionCol="prediction", metricName="rmse",)
rmse_nv = evaluator_nv.evaluate(pred_nv)

# Calculate r2 for non violent crimes
evaluator2_nv = RegressionEvaluator(labelCol="num_crimes", predictionCol="prediction", metricName="r2",)
r2_nv = evaluator2_nv.evaluate(pred_nv)

with open('3_Q.txt', 'w') as f:
	f.write("For the  model that predicts the number of violent crimes at a week-beat level for the next week:\n")
	f.write("R squared: ")
	f.write(str(r2_v))
	f.write("\n")
	f.write("RMSE: ")
	f.write(str(rmse_v))
	f.write("\n")
	f.write("\n")
	f.write("For the model that predicts the number of non-violent crimes at a week-beat level for the next week:\n")
	f.write("R squared: ")
	f.write(str(r2_nv))
	f.write("\n")
	f.write("RMSE: ")
	f.write(str(rmse_nv))
