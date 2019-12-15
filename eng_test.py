import os
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import DCT
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import StandardScaler

from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

def init_spark():
  spark = SparkSession.builder.appName("eng_test").config('spark.driver.memory', '4g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()


df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_stat = spark.sql("""
SELECT seg,
    AVG(x) AS x_avg,
    STD(x)AS x_std, 
    SKEWNESS(x) AS x_skew, 
    KURTOSIS(x) AS x_kurt, 
    MAX(x) AS x_max,
    MIN(x) AS x_min,
    MAX(ABS(x)) AS xa_max,
    MIN(ABS(x)) AS xa_min,
    PERCENTILE(x, 0.05) AS x_p1,
    PERCENTILE(x, 0.20) AS x_p2,
    PERCENTILE(x, 0.50) AS x_p5,
    PERCENTILE(x, 0.80) AS x_p8,
    PERCENTILE(x, 0.95) AS x_p9,
    AVG(ABS(x)) AS xa_avg,
    STD(ABS(x))AS xa_std, 
    SKEWNESS(ABS(x)) AS xa_skew, 
    KURTOSIS(ABS(x)) AS xa_kurt, 
    PERCENTILE(ABS(x), 0.05) AS xa_p1,
    PERCENTILE(ABS(x), 0.20) AS xa_p2,
    PERCENTILE(ABS(x), 0.50) AS xa_p5,
    PERCENTILE(ABS(x), 0.80) AS xa_p8,
    PERCENTILE(ABS(x), 0.95) AS xa_p9,
    PERCENTILE(x, 0.01) AS x_p01,
    PERCENTILE(x, 0.02) AS x_p02,
    PERCENTILE(x, 0.05) AS x_p05,
    PERCENTILE(x, 0.10) AS x_p10,
    PERCENTILE(x, 0.90) AS x_p90,
    PERCENTILE(x, 0.95) AS x_p95,
    PERCENTILE(x, 0.98) AS x_p98,
    PERCENTILE(x, 0.99) AS x_p99,
    PERCENTILE(ABS(x), 0.90) AS xa_p90,
    PERCENTILE(ABS(x), 0.92) AS xa_p92,
    PERCENTILE(ABS(x), 0.95) AS xa_p95,
    PERCENTILE(ABS(x), 0.98) AS xa_p98,
    PERCENTILE(ABS(x), 0.99) AS xa_p99,
    (MAX(x) - MIN(x)) AS x_range
FROM data 
GROUP BY seg
ORDER BY seg
""" )

from pyspark.ml.feature import VectorAssembler
cols = ["x_avg", "x_std", "x_skew", "x_kurt", "x_max", "x_min", "xa_max", "xa_min", "x_p1", "x_p2", "x_p5", "x_p8", "x_p9",
        "xa_avg", "xa_std", "xa_skew", "xa_kurt", "xa_p1", "xa_p2", "xa_p5", "xa_p8", "xa_p9",
        "x_p01","x_p02","x_p05","x_p10","x_p90","x_p95","x_p98","x_p99",
        "xa_p90","xa_p92","xa_p95","xa_p98","xa_p99", "x_range"]
assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
df_stat = assembler.transform(df_stat)
for col in cols: df_stat = df_stat.drop(col)


scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
scalerModel = scaler.fit(df_stat)
df_stat = scalerModel.transform(df_stat).drop("stat_t1")

df_stat.write.mode("overwrite").parquet(os.path.join("data", "train.stat.3.parquet"))