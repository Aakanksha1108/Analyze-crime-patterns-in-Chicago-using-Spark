#Spark HW - Question Number 2a

# Importing packages
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
import csv

# Step 1. Establish connection
sc = SparkContext()

# Step 2. Read data
crime = sc.textFile('hdfs://wolf.analytics.private/user/asp9295/Assignment_Spark/Crimes_-_2001_to_present.csv')
# Remove header -> Source: https://stackoverflow.com/questions/27854919/how-do-i-skip-a-header-from-csv-files-in-spark
header = crime.first() #extract header

# Step 3. Manipulate data. Here, I have considered the entire block value (not just the first 5 characters) and filtered information for the last 3 years (2017 to 2019)

crime_filtered_data = crime.filter(lambda row: row != header).map(lambda row: row.split(",")).map(lambda row: (row[17],row[3])).filter(lambda row: row[0]=="2017" or row[0]=="2018" or row[0]=="2019").map(lambda row: (row[1],1))
top_blocks = crime_filtered_data.reduceByKey(lambda x,y:x+y).takeOrdered(10, key= lambda x: -x[1])

#write output to text
with open("Sah_exercise2a.txt", 'w') as f:
    writer = csv.writer(f)
    f.write("Block \t Number of crimes\n")
    for line in top_blocks:
    	writer.writerow(line)
