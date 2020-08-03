
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import date_format
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import from_unixtime
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import numpy as np
import pandas as pd

sc = SparkContext()
sqlcontext = SQLContext(sc)

# Step 1: Reading data
path = 'hdfs://wolf.analytics.private/user/asp9295/Assignment_Spark/Crimes_-_2001_to_present.csv'
crime = sqlcontext.read.csv(path, header = True)

# Step 2: Filter for arrests
crime = crime.filter(crime.Arrest==True)

# Step 3: Creating additional relevant features
crime = crime.withColumn('day_of_week', date_format(to_date(crime['Date'], "MM/dd/yyyy"), 'u').cast(IntegerType()))
crime = crime.withColumn('time_part', split(crime['Date']," ").getItem(1))
crime = crime.withColumn('time_of_day',split(crime['time_part'],":").getItem(0).cast(IntegerType()))
crime = crime.withColumn('AM/PM', substring(crime['Date'],-2,2))
crime = crime.withColumn('date_part', split(crime['Date']," ").getItem(0))
crime = crime.withColumn('month',split(crime['Date'], '/').getItem(0).cast(IntegerType()))
crime = crime.withColumn("time_of_day_v2", when((crime['AM/PM']=='PM'),crime['time_of_day']+12).otherwise(crime['time_of_day']))

# Step 4: Calculating summaries
month_level = crime.groupBy("month","Year").count()
month_level = month_level.groupBy("month").mean("count").orderBy("month")
month_level = month_level.toPandas().set_index("month")

#Step 5: plots
month_level.plot.bar(figsize=(12,10))
plt.title("Average Arrests - Monthly Level")
plt.ylabel("Number of Arrests")
plt.savefig("Sah_exercise4_month.png")
plt.figure()

dow_level = crime.groupBy("day_of_week","date_part").count()
dow_level = dow_level.groupBy("day_of_week").mean("count").orderBy("day_of_week")
dow_level = dow_level.toPandas().set_index("day_of_week")

dow_level.plot.bar(figsize=(12,10))
plt.title("Average Arrests - Day of Week Level")
plt.ylabel("Number of Arrests")
plt.savefig("Sah_exercise4_dow.png")
plt.figure()

tod_level = crime.groupBy("time_of_day_v2","date_part").count()
tod_level = tod_level.groupBy("time_of_day_v2").mean("count").orderBy("time_of_day_v2")
tod_level = tod_level.toPandas().set_index("time_of_day_v2")

tod_level.plot.bar(figsize=(12,10))
plt.title("Average Arrests - Hourly Level")
plt.ylabel("Number of Arrests")
plt.savefig("Sah_exercise4_tod.png")
plt.figure()
