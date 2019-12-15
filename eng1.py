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
  spark = SparkSession.builder.appName("eng1").config('spark.driver.memory', '8g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()


def slice_win_source_to(source, destination):
    df_w = spark.read.parquet(os.path.join("data", source))
    for j in range(8):
        slicer = VectorSlicer(inputCol="f" + str(j), outputCol="f_sl" + str(j), indices=[i for i in range(43, 99)])
        df_w = slicer.transform(df_w).drop("f" + str(j))
    cols = ["f_sl" + str(i) for i in range(8)]
    assembler = VectorAssembler(inputCols=cols, outputCol="f")
    df_w = assembler.transform(df_w)
    df_w.write.mode("overwrite").parquet(os.path.join("data", destination))
    df_w.printSchema()


slice_win_source_to("train.win.vector.fbin.parquet", "train.win.vector.fbin.3.parquet")
slice_win_source_to("test.win.vector.fbin.parquet", "test.win.vector.fbin.3.parquet")


















