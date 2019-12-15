from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import monotonically_increasing_id, row_number, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as fn
import os

from pyspark.sql import SparkSession
import pandas as pd
import shutil
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark import SparkContext



def init_spark():
  spark = SparkSession.builder.appName("ETL").config('spark.driver.memory', '8g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()

schema = StructType([
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True)])

"""
w1 = Window.orderBy("uid")
w2 = Window.partitionBy("seg").orderBy("uid")
df_train = spark.read.csv(os.path.join("data","train.csv"), header=True,schema=schema).withColumn(
    "uid", monotonically_increasing_id()).withColumn(
    "idx", row_number().over(w1).cast(IntegerType())).withColumn(
    "seg", fn.floor(((fn.col("idx")-1)/150000)).cast(IntegerType())).withColumn(
    "no", row_number().over(w2).cast(IntegerType())).withColumn(
    "name", fn.concat(lit("raw_"),fn.lpad(fn.col("seg"),4,"0").cast(StringType()))).withColumn(
    "set", lit(0))

df_train.createOrReplaceTempView("data")
df_train_f = spark.sql("""
#SELECT uid, set, seg, no, name, x, y FROM data
#ORDER BY set, seg, no, uid
""")

df_train_f = df_train_f.repartition(1)
df_train_f.write.mode("overwrite").parquet(os.path.join("data", "train.parquet"))

"""
"""
df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
max_id = spark.sql("""
#SELECT max(uid) as m FROM data
""").first().m
print("max_id", max_id)

df_result = pd.read_csv(os.path.join("data", "sample_submission.csv"))
files = list(df_result["seg_id"].values)

schema = StructType([StructField("x", DoubleType(), True)])

seg = 0
for file in files:
    sep = "."
    if seg % 200 == 0: sep = "|"
    if seg % 20 == 0: print(sep, end="", flush=True)
    seg += 1
print("", end="\n", flush=True)

seg = 0
df_test = None
for file in files:
    #     print(file)
    if seg % 20 == 0: print("|", end="", flush=True)

    w1 = Window.orderBy("uid")
    w2 = Window.partitionBy("seg").orderBy("uid")
    df_temp = spark.read.csv(os.path.join("data", "test", file + ".csv"), header=True, schema=schema).withColumn(
        "y", lit(None).cast(DoubleType())).withColumn(
        "uid", lit(max_id + 1) + monotonically_increasing_id()).withColumn(
        "idx", row_number().over(w1).cast(IntegerType())).withColumn(
        "seg", lit(seg).cast(IntegerType())).withColumn(
        "no", row_number().over(w2).cast(IntegerType())).withColumn(
        "name", (lit(file.split(".")[0])).cast(StringType())).withColumn(
        "set", lit(1))

    df_temp.createOrReplaceTempView("data")
    df_temp_f = spark.sql("""
#    SELECT uid, set, seg, no, name, x, y FROM data
#    ORDER BY set, seg, no, uid
#    """)

#    max_id = spark.sql("""
#    SELECT max(uid) as m FROM data
#   """).first().m
"""
    seg += 1

    if df_test == None:
        df_test = df_temp_f
    else:
        df_test = df_test.union(df_temp_f)

    # create 1 file per 20 = I had issue when processing all in one go
    if seg % 20 == 0:
        file_name = "test_1_{:04}.parquet".format(seg)
        df_test = df_test.repartition(1)
        if os.path.isdir(os.path.join("data", file_name)):
            df_test = None
            continue
        df_test.write.parquet(os.path.join("data", file_name))
        df_test = None
#     if seg == 4 : break

print("(", end="", flush=True)
# left under 20 batch
if df_test != None:
    file_name = "test_1_{:04}.parquet".format(seg)
    df_test = df_test.repartition(1)
    df_test.write.parquet(os.path.join("data", file_name))
    df_test = None
print("x)", end="\n", flush=True)

print("max_id", max_id)


"""
df_result = pd.read_csv(os.path.join("data", "sample_submission.csv"))
files = list(df_result["seg_id"].values)

seg = 0
for file in files:
    sep = "."
    if seg % 200 == 0: sep = "|"
    if seg % 20 == 0: print(sep, end="", flush=True)
    seg += 1
print("", end="\n", flush=True)

seg = 0
mode = "overwrite"
for file in files:
    if seg % 20 == 0: print("|", end="", flush=True)
    seg += 1

    if seg % 20 == 0:
        file_name = "test_1_{:04}.parquet".format(seg)
        df_test = spark.read.parquet(os.path.join("data", file_name))
        df_test.write.mode(mode).parquet(os.path.join("data", "test.parquet"))
        mode = "append"

print("(", end="", flush=True)

# left under 20 batch
if seg % 20 != 0 :
    file_name = "test_1_{:04}.parquet".format(seg)
    df_test = spark.read.parquet(os.path.join("data", file_name))
    df_test.write.mode(mode).parquet(os.path.join("data", "test.parquet"))
    mode = "append"

print("x", end="", flush=True)

print(")", end="\n", flush=True)

df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_test = spark.read.parquet(os.path.join("data", "test.parquet"))

for df in [df_train, df_test]:
    df.createOrReplaceTempView("data")
    nb_seg = spark.sql("""
    SELECT set, seg, COUNT(*) AS n FROM data GROUP BY set, seg
    """)

    pddf = nb_seg.toDF("set", "seg", "n").toPandas()

    print("shape", pddf.shape)
    print("!= 150 000",pddf[pddf["n"]!=150000])

#30 sec to perform a calculation on the whole dataset, that sounds good

seg = 0
for file in files:
    sep = "."
    if seg % 200 == 0: sep = "|"
    if seg % 20 == 0: print(sep, end="", flush=True)
    seg += 1
print("", end="\n", flush=True)

seg = 0
for file in files:
    if seg % 20 == 0: print("|", end="", flush=True)
    seg += 1

    if seg % 20 == 0:
        file_name = "test_1_{:04}.parquet".format(seg)
        shutil.rmtree(os.path.join("data", file_name))

print("(", end="", flush=True)
# left under 20 batch
if seg % 20 != 0 :
    file_name = "test_1_{:04}.parquet".format(seg)
    shutil.rmtree(os.path.join("data", file_name))
print("x", end="", flush=True)
print(")", end="\n", flush=True)

df = spark.read.parquet(os.path.join("data", "train.parquet"))
df.createOrReplaceTempView("data")


df_agg = sc.parallelize([])
for i in range(10):
    sql = """
    SELECT seg, x FROM data WHERE set = 0 AND no BETWEEN {0} AND {1} ORDER BY uid
    """.format(i*15000, (i+1)*15000)
    df_temp = spark.sql(sql)
    rdd_temp = df_temp.rdd.map(lambda row:(row.seg,row.x))      \
        .map(lambda data: (data[0], [ data[1] ]))               \
        .reduceByKey(lambda a, b: a + b)                        \
        .map(lambda row: Row(seg=row[0],vx=Vectors.dense(row[1])))
    if df_agg.count() == 0:
        df_agg = rdd_temp.toDF(["seg","vx"+str(i)])
    else:
        df_temp = rdd_temp.toDF(["seg0","vx"+str(i)])
        df_agg = df_agg.join(df_temp, df_agg.seg == df_temp.seg0).drop("seg0")

df_agg.write.mode("overwrite").parquet(os.path.join("data", "train.vector.parquet"))


df = spark.read.parquet(os.path.join("data", "test.parquet"))
df.createOrReplaceTempView("data")


df_agg = sc.parallelize([])
for i in range(10):
    sql = """
    SELECT seg, x FROM data WHERE set = 1 AND no BETWEEN {0} AND {1} ORDER BY uid
    """.format(i*15000, (i+1)*15000)
    df_temp = spark.sql(sql)
    rdd_temp = df_temp.rdd.map(lambda row:(row.seg,row.x))      \
        .map(lambda data: (data[0], [ data[1] ]))               \
        .reduceByKey(lambda a, b: a + b)                        \
        .map(lambda row: Row(seg=row[0],vx=Vectors.dense(row[1])))
    if df_agg.count() == 0:
        df_agg = rdd_temp.toDF(["seg","vx"+str(i)])
    else:
        df_temp = rdd_temp.toDF(["seg0","vx"+str(i)])
        df_agg = df_agg.join(df_temp, df_agg.seg == df_temp.seg0).drop("seg0")

df_agg.write.mode("overwrite").parquet(os.path.join("data", "test.vector.parquet"))