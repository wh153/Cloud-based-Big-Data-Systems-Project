# Databricks notebook source
# MAGIC %md
# MAGIC # Cloud-based-Big-Data-Systems-Project

# COMMAND ----------

# load data
df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/wh153@duke.edu/Test.csv")
df2 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/wh153@duke.edu/Train.csv")

# COMMAND ----------

display(df1)

# COMMAND ----------

display(df2)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the packages

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download('all')

# COMMAND ----------

# label convertion and text cleaning
def label_convert(r):
    """
    Convert string lables into integer so that it could be processed by machine learning algortihms
    """
    if r == "Depression":
        return 0
    elif r == "Drugs":
        return 1
    elif r == "Alcohol":
        return 2
    elif r == "Suicide":
        return 3
    
def clean_text(r):
    """
    clean text
    """
    nltk.download('stopwords')
    r = r.lower()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    r_tokens = tokenizer.tokenize(r)
    stop_words = set(stopwords.words('english'))
    filtered_r = [w for w in r_tokens if not w in stop_words]
    return " ".join(filtered_r)

# COMMAND ----------

test = df1
train = df2

# COMMAND ----------

display(train.groupby('label').count())

# COMMAND ----------

# define udf to map label_convert helper function to dataframe
udf_label_convert = F.udf(label_convert, IntegerType())
train= train.withColumn("labelnum", udf_label_convert("label"))
train.show()


# COMMAND ----------

# define udf to map label_convert helper function to dataframe
udf_clean_text = F.udf(clean_text, StringType())
train = train.withColumn("cleaned_text", udf_clean_text("text"))
train.show()

# COMMAND ----------

pd_train = train.toPandas()
pd_train

# COMMAND ----------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(pd_train['cleaned_text'])
Y = pd_train['labelnum']

# COMMAND ----------

clf = LogisticRegression(random_state=0).fit(X, Y)
clf.predict(X)
clf.score(X, Y)

# COMMAND ----------


