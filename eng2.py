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
  spark = SparkSession.builder.appName("eng2").config('spark.driver.memory', '4g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()

df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_stat = spark.sql("""
SELECT seg,
    AVG(x) AS x_avg,
    STD(x) AS x_std,
    SKEWNESS(x) AS x_skew,
    KURTOSIS(x) AS x_kurt, 
    MAX(x) AS x_max,
    MIN(x) AS x_min,MIN(x) AS x_min,
    PERCENTILE(x, 0.01) AS x_p01,
    AVG(PERCENTILE(x, 0.01)) AS xp01_avg,
    STD(PERCENTILE(x, 0.01)) AS xp01_std,
    SKEWNESS(PERCENTILE(x, 0.01)) AS xp01_skew,
    KURTOSIS(PERCENTILE(x, 0.01)) AS xp01_kurt, 
    MAX(PERCENTILE(x, 0.01)) AS xp01_max,
    MIN(PERCENTILE(x, 0.01)) AS xp01_min,
    PERCENTILE(x, 0.02) AS x_p02,
    AVG(PERCENTILE(x, 0.02)) AS xp02_avg,
    STD(PERCENTILE(x, 0.02)) AS xp02_std,
    SKEWNESS(PERCENTILE(x, 0.02)) AS xp02_skew,
    KURTOSIS(PERCENTILE(x, 0.02)) AS xp02_kurt, 
    MAX(PERCENTILE(x, 0.02)) AS xp02_max,
    MIN(PERCENTILE(x, 0.02)) AS xp02_min,
    PERCENTILE(x, 0.05) AS x_p05,
    AVG(PERCENTILE(x, 0.05)) AS xp05_avg,
    STD(PERCENTILE(x, 0.05)) AS xp05_std,
    SKEWNESS(PERCENTILE(x, 0.05)) AS xp05_skew,
    KURTOSIS(PERCENTILE(x, 0.05)) AS xp05_kurt, 
    MAX(PERCENTILE(x, 0.05)) AS xp05_max,
    MIN(PERCENTILE(x, 0.05)) AS xp05_min,
    PERCENTILE(x, 0.10) AS x_p10,
    AVG(PERCENTILE(x, 0.10)) AS xp10_avg,
    STD(PERCENTILE(x, 0.10)) AS xp10_std,
    SKEWNESS(PERCENTILE(x, 0.10)) AS xp10_skew,
    KURTOSIS(PERCENTILE(x, 0.10)) AS xp10_kurt, 
    MAX(PERCENTILE(x, 0.10)) AS xp10_max,
    MIN(PERCENTILE(x, 0.10)) AS xp10_min,
    PERCENTILE(x, 0.90) AS x_p90,
    AVG(PERCENTILE(x, 0.90)) AS xp90_avg,
    STD(PERCENTILE(x, 0.90)) AS xp90_std,
    SKEWNESS(PERCENTILE(x, 0.90)) AS xp90_skew,
    KURTOSIS(PERCENTILE(x, 0.90)) AS xp90_kurt, 
    MAX(PERCENTILE(x, 0.90)) AS xp90_max,
    MIN(PERCENTILE(x, 0.90)) AS xp90_min,
    PERCENTILE(x, 0.95) AS x_p95,
    AVG(PERCENTILE(x, 0.95)) AS xp95_avg,
    STD(PERCENTILE(x, 0.95)) AS xp95_std,
    SKEWNESS(PERCENTILE(x, 0.95)) AS xp95_skew,
    KURTOSIS(PERCENTILE(x, 0.95)) AS xp95_kurt, 
    MAX(PERCENTILE(x, 0.95)) AS xp95_max,
    MIN(PERCENTILE(x, 0.95)) AS xp95_min,
    PERCENTILE(x, 0.98) AS x_p98,
    AVG(PERCENTILE(x, 0.98)) AS xp98_avg,
    STD(PERCENTILE(x, 0.98)) AS xp98_std,
    SKEWNESS(PERCENTILE(x, 0.98)) AS xp98_skew,
    KURTOSIS(PERCENTILE(x, 0.98)) AS xp98_kurt, 
    MAX(PERCENTILE(x, 0.98)) AS xp98_max,
    MIN(PERCENTILE(x, 0.98)) AS xp98_min,
    PERCENTILE(x, 0.99) AS x_p99,
    AVG(PERCENTILE(x, 0.99)) AS xp99_avg,
    STD(PERCENTILE(x, 0.99)) AS xp99_std,
    SKEWNESS(PERCENTILE(x, 0.99)) AS xp99_skew,
    KURTOSIS(PERCENTILE(x, 0.99)) AS xp99_kurt, 
    MAX(PERCENTILE(x, 0.99)) AS xp99_max,
    MIN(PERCENTILE(x, 0.99)) AS xp99_min,
    PERCENTILE(ABS(x), 0.90) AS xa_p90,
    AVG(PERCENTILE(ABS(x), 0.90)) AS xap90_avg,
    STD(PERCENTILE(ABS(x), 0.90)) AS xap90_std,
    SKEWNESS(PERCENTILE(ABS(x), 0.90)) AS xap90_skew,
    KURTOSIS(PERCENTILE(ABS(x), 0.90)) AS xap90_kurt, 
    MAX(PERCENTILE(ABS(x), 0.90)) AS xap90_max,
    MIN(PERCENTILE(ABS(x), 0.90)) AS xap90_min,
    PERCENTILE(ABS(x), 0.92) AS xa_p92,
    AVG(PERCENTILE(ABS(x), 0.92)) AS xap92_avg,
    STD(PERCENTILE(ABS(x), 0.92)) AS xap92_std,
    SKEWNESS(PERCENTILE(ABS(x), 0.92)) AS xap92_skew,
    KURTOSIS(PERCENTILE(ABS(x), 0.92)) AS xap92_kurt, 
    MAX(PERCENTILE(ABS(x), 0.92)) AS xap92_max,
    MIN(PERCENTILE(ABS(x), 0.92)) AS xap92_min,
    PERCENTILE(ABS(x), 0.95) AS xa_p95,
    AVG(PERCENTILE(ABS(x), 0.95)) AS xap95_avg,
    STD(PERCENTILE(ABS(x), 0.95)) AS xap95_std,
    SKEWNESS(PERCENTILE(ABS(x), 0.95)) AS xap95_skew,
    KURTOSIS(PERCENTILE(ABS(x), 0.95)) AS xap95_kurt, 
    MAX(PERCENTILE(ABS(x), 0.95)) AS xap95_max,
    MIN(PERCENTILE(ABS(x), 0.95)) AS xap95_min,
    PERCENTILE(ABS(x), 0.98) AS xa_p98,
    AVG(PERCENTILE(ABS(x), 0.98)) AS xap98_avg,
    STD(PERCENTILE(ABS(x), 0.98)) AS xap98_std,
    SKEWNESS(PERCENTILE(ABS(x), 0.98)) AS xap98_skew,
    KURTOSIS(PERCENTILE(ABS(x), 0.98)) AS xap98_kurt, 
    MAX(PERCENTILE(ABS(x), 0.98)) AS xap98_max,
    MIN(PERCENTILE(ABS(x), 0.98)) AS xap98_min,
    PERCENTILE(ABS(x), 0.99) AS xa_p99,
    AVG(PERCENTILE(ABS(x), 0.99)) AS xap99_avg,
    STD(PERCENTILE(ABS(x), 0.99)) AS xap99_std,
    SKEWNESS(PERCENTILE(ABS(x), 0.99)) AS xap99_skew,
    KURTOSIS(PERCENTILE(ABS(x), 0.99)) AS xap99_kurt, 
    MAX(PERCENTILE(ABS(x), 0.99)) AS xap99_max,
    MIN(PERCENTILE(ABS(x), 0.99)) AS xap99_min,
    MAX(STD(PERCENTILE(ABS(x), 0.90)), STD(PERCENTILE(ABS(x), 0.92)), STD(PERCENTILE(ABS(x), 0.95)), STD(PERCENTILE(ABS(x), 0.98)), STD(PERCENTILE(ABS(x), 0.99))) AS xaSTD_max,
    AVG(STD(PERCENTILE(ABS(x), 0.90)), STD(PERCENTILE(ABS(x), 0.92)), STD(PERCENTILE(ABS(x), 0.95)), STD(PERCENTILE(ABS(x), 0.98)), STD(PERCENTILE(ABS(x), 0.99))) AS xaAVG,
FROM data 
GROUP BY seg
ORDER BY seg
""" )

from pyspark.ml.feature import VectorAssembler
cols = ["x_avg", "x_p01","x_p02","x_p05","x_p10","x_p90","x_p95","x_p98","x_p99",
        "xa_p90","xa_p92","xa_p95","xa_p98","xa_p99", "x_std", "x_skew", "x_kurt", "x_max", "x_min", "xp01_avg", "xp01_std", "xp01_skew"
        "xp01_kurt", "xp01_max", "xp01_min", "xp02_avg", "xp02_std", "xp02_skew"
        "xp02_kurt", "xp02_max", "xp02_min", "xp05_avg", "xp05_std", "xp05_skew"
        "xp05_kurt", "xp05_max", "xp05_min", "xp10_avg", "xp10_std", "xp10_skew"
        "xp10_kurt", "xp10_max", "xp10_min", "xp90_avg", "xp90_std", "xp90_skew"
        "xp90_kurt", "xp90_max", "xp90_min", "xp95_avg", "xp95_std", "xp95_skew"
        "xp95_kurt", "xp95_max", "xp95_min", "xp98_avg", "xp98_std", "xp98_skew"
        "xp98_kurt", "xp98_max", "xp98_min", "xp99_avg", "xp99_std", "xp99_skew"
        "xp99_kurt", "xp99_max", "xp99_min", "xap90_avg", "xap90_std", "xap90_skew"
        "xap90_kurt", "xap90_max", "xap90_min", "xap92_avg", "xap92_std", "xap92_skew"
        "xap92_kurt", "xap92_max", "xap92_min", "xap95_avg", "xap95_std", "xap95_skew"
        "xap95_kurt", "xap95_max", "xap95_min", "xap98_avg", "xap98_std", "xap98_skew"
        "xap98_kurt", "xap98_max", "xap98_min", "xap99_avg", "xap99_std", "xap99_skew"
        "xap99_kurt", "xap99_max", "xap99_min", "xaSTD_max", "xaAVG"]
assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
df_stat = assembler.transform(df_stat)
for col in cols: df_stat = df_stat.drop(col)

from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
scalerModel = scaler.fit(df_stat)
df_stat = scalerModel.transform(df_stat).drop("stat_t1")

df_stat.write.mode("overwrite").parquet(os.path.join("data", "train.stat.3.parquet"))