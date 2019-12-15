import os
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
import matplotlib.pyplot as plt


def init_spark():
  spark = SparkSession.builder.appName("model2").config('spark.driver.memory', '4g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()

df_f = spark.read.parquet(os.path.join("data", "train.vector.fbin.2.parquet"))
# df_s = spark.read.parquet(os.path.join("data", "train.stat.parquet"))
df_s = spark.read.parquet(os.path.join("data", "train.stat.3.parquet"))
df_y = spark.read.parquet(os.path.join("data", "train.target.parquet"))

df_f = df_f.selectExpr("*").drop("_c0")
df_s = df_s.selectExpr("seg AS seg1", "*").drop("seg").drop("_c0")
df_y = df_y.selectExpr("seg AS seg2", "y AS label")

df_train = df_f
df_train = df_train.join(df_s, df_train.seg.cast(IntegerType()) == df_s.seg1.cast(IntegerType())).drop("seg1")
df_train = df_train.join(df_y, df_train.seg.cast(IntegerType()) == df_y.seg2.cast(IntegerType())).drop("seg2")

df_train.printSchema()

vect_cols = ["f"]
vectorAssembler = VectorAssembler(inputCols=vect_cols, outputCol="features")

df_train = vectorAssembler.transform(df_train)

trainingData = df_train.selectExpr("*").where("seg < 3145")
testData = df_train.selectExpr("*").where("seg >= 3145")


evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    numTrees=200,
    maxDepth=30,
    maxBins=32,
    minInstancesPerNode=1,
    minInfoGain=0.0,
    maxMemoryInMB=256,
    cacheNodeIds=False,
    checkpointInterval=10,
    impurity='variance',
    subsamplingRate=1.0,
    seed=None,
    featureSubsetStrategy='auto')

# from pyspark.ml import Pipeline
# pipeline = Pipeline(stages=[vectorAssembler, forest])

model_rf = rf.fit(trainingData)
predictions_rf = model_rf.transform(testData)

model_rf.write().overwrite().save("model.1.2.spark.rf.data")

mae = evaluator.evaluate(predictions_rf)
print("mae on test data with random forest model = {:.8f} ".format(mae))

pd_pred = predictions_rf.orderBy("seg").toPandas()
fig, ax = plt.subplots(figsize=(20,8))
plt.plot(pd_pred["prediction"], alpha=0.8, label="predictions")
plt.plot(pd_pred["label"], label="labels")
title = ax.set_title("Prediction vs. ground truth", loc="left")
legend = plt.legend(loc="upper left")
plt.savefig("randomForest2.png")

