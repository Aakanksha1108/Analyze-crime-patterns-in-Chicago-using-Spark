#Spark HW - Question Number 1

# Importing packages

from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1. Establish connection
sc = SparkContext()
sqlcontext = SQLContext(sc)

# Step 2. Read dataset

path = 'hdfs://wolf.analytics.private/user/asp9295/Assignment_Spark/Crimes_-_2001_to_present.csv'
crime = sqlcontext.read.csv(path, header = True)

# Step 3. Perform necessary data transformations
crime = crime.withColumn('Day_Part', split(crime['Date'], ' ').getItem(0))
crime = crime.withColumn('Month_Part', split(crime['Day_Part'], '/').getItem(0))
crime = crime.withColumn('Year_Part', split(crime['Day_Part'], '/').getItem(2))
crime.createOrReplaceTempView('Data_Q1')

# Step 4. Sql query to roll up the data
rolled_up_data = sqlcontext.sql("SELECT DISTINCT Month_Part, COUNT(*) / COUNT(DISTINCT Year_Part) AS Average_Monthly_Crime from Data_Q1 group by Month_Part order by Month_Part").toPandas()

# Step 5. Create bar plot
plt.bar(rolled_up_data['Month_Part'], rolled_up_data['Average_Monthly_Crime'], color=(0.2, 0.4, 0.6, 0.6))
plt.title('Crime events in Chicago seem to be higher in the months of July & August')
plt.xticks(np.arange(12),('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
plt.ylabel('Average number of crime events')
plt.savefig('Sah_exercise1.png')
