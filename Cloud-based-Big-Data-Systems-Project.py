# Databricks notebook source

df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/wh153@duke.edu/Test.csv")
df2 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/wh153@duke.edu/Train.csv")

# COMMAND ----------


