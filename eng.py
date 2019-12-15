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
  spark = SparkSession.builder.appName("eng").config('spark.driver.memory', '4g').getOrCreate()

  sc = spark.sparkContext
  return spark,sc

spark,sc = init_spark()

df_train = spark.read.parquet(os.path.join("data", "train.vector.parquet"))
df_test = spark.read.parquet(os.path.join("data", "test.vector.parquet"))

cols = ["vx"+str(i) for i in range(10)]
assembler = VectorAssembler(inputCols=cols, outputCol="vx_t1")
dct = DCT(inverse=False, inputCol="vx_t1", outputCol="vx_t2")
slicer = VectorSlicer(inputCol="vx_t2", outputCol="vx_t3", indices=[i for i in range(40000)])
scaler = StandardScaler(inputCol="vx_t3", outputCol="vx", withStd=True, withMean=False)

pipeline = Pipeline(stages=[assembler, dct, slicer, scaler])
p_model = pipeline.fit(df_train)

df_train = p_model.transform(df_train)
df_train = df_train.drop("vx0").drop("vx1").drop("vx2").drop("vx3").drop("vx4")
df_train = df_train.drop("vx5").drop("vx6").drop("vx7").drop("vx8").drop("vx9")
df_train = df_train.drop("vx_t1").drop("vx_t2").drop("vx_t3")

df_test = p_model.transform(df_test)
df_test = df_test.drop("vx0").drop("vx1").drop("vx2").drop("vx3").drop("vx4")
df_test = df_test.drop("vx5").drop("vx6").drop("vx7").drop("vx8").drop("vx9")
df_test = df_test.drop("vx_t1").drop("vx_t2").drop("vx_t3")

df_train.write.mode("overwrite").parquet(os.path.join("data", "train.vector.dct.parquet"))
df_test.write.mode("overwrite").parquet(os.path.join("data", "test.vector.dct.parquet"))

df_vect = spark.read.parquet(os.path.join("data", "test.vector.dct.parquet"))


def bin_source_to(source, destination):
    df_v = spark.read.parquet(os.path.join("data", source))
    rdd_v = df_v.rdd.map(lambda row: (row.seg, row.vx, []))
    for i in range(0, 99):
        rdd_v = rdd_v.map(lambda row, i=i: (
        row[0], row[1], row[2] + [float(sum(abs(row[1].toArray()[i * 400:i * 400 + 800])) / 800)]))
    rdd_v = rdd_v.map(lambda row: Row(seg=row[0], f=Vectors.dense(row[2])))
    df_v = rdd_v.toDF()
    df_v.write.mode("overwrite").parquet(os.path.join("data", destination))


bin_source_to("train.vector.dct.parquet", "train.vector.fbin.parquet")
bin_source_to("test.vector.dct.parquet", "test.vector.fbin.parquet")

slicer = VectorSlicer(inputCol="f_t1", outputCol="f", indices=[i for i in range(50, 76)])


def slice_source_to(source, destination):
    df_v = spark.read.parquet(os.path.join("data", source))
    df_v = df_v.selectExpr("*", "f AS f_t1").drop("_c0").drop("f")
    df_v = slicer.transform(df_v).drop("f_t1")
    df_v.write.mode("overwrite").parquet(os.path.join("data", destination))


slice_source_to("train.vector.fbin.parquet", "train.vector.fbin.2.parquet")
slice_source_to("test.vector.fbin.parquet", "test.vector.fbin.2.parquet")



df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_stat = spark.sql("""
SELECT seg,
    AVG(x) AS x_avg,
    STD(x)AS x_std, 
    SKEWNESS(x) AS x_skew, 
    KURTOSIS(x) AS x_kurt, 
    MAX(x) AS x_max,
    PERCENTILE(x, 0.05) AS x_p1,
    PERCENTILE(x, 0.20) AS x_p2,
    PERCENTILE(x, 0.50) AS x_p5,
    PERCENTILE(x, 0.80) AS x_p8,
    PERCENTILE(x, 0.95) AS x_p9,
    AVG(ABS(x)) AS xa_avg,
    STD(ABS(x))AS xa_std, 
    SKEWNESS(ABS(x)) AS xa_skew, 
    KURTOSIS(ABS(x)) AS xa_kurt, 
    MAX(ABS(x)) AS xa_max,
    PERCENTILE(ABS(x), 0.05) AS xa_p1,
    PERCENTILE(ABS(x), 0.20) AS xa_p2,
    PERCENTILE(ABS(x), 0.50) AS xa_p5,
    PERCENTILE(ABS(x), 0.80) AS xa_p8,
    PERCENTILE(ABS(x), 0.95) AS xa_p9
FROM data 
GROUP BY seg
ORDER BY seg
""" )

cols = ["x_avg", "x_std", "x_skew", "x_kurt", "x_max", "x_p1", "x_p2", "x_p5", "x_p8", "x_p9",
        "xa_avg", "xa_std", "xa_skew", "xa_kurt", "xa_max", "xa_p1", "xa_p2", "xa_p5", "xa_p8", "xa_p9"]
assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
df_stat = assembler.transform(df_stat)
for col in cols: df_stat = df_stat.drop(col)


scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
scalerModel = scaler.fit(df_stat)
df_stat = scalerModel.transform(df_stat).drop("stat_t1")

df_stat.write.mode("overwrite").parquet(os.path.join("data", "train.stat.parquet"))

df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_stat = spark.sql("""
SELECT seg,
    AVG(x) AS x_avg,
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
    PERCENTILE(ABS(x), 0.99) AS xa_p99
FROM data 
GROUP BY seg
ORDER BY seg
""" )

cols = ["x_avg", "x_p01","x_p02","x_p05","x_p10","x_p90","x_p95","x_p98","x_p99",
        "xa_p90","xa_p92","xa_p95","xa_p98","xa_p99"]
assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
df_stat = assembler.transform(df_stat)
for col in cols: df_stat = df_stat.drop(col)


scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
scalerModel = scaler.fit(df_stat)
df_stat = scalerModel.transform(df_stat).drop("stat_t1")

df_stat.write.mode("overwrite").parquet(os.path.join("data", "train.stat.2.parquet"))

df_test = spark.read.parquet(os.path.join("data", "test.parquet"))
df_test.createOrReplaceTempView("data")
df_stat = spark.sql("""
SELECT seg,
    AVG(x) AS x_avg,
    STD(x)AS x_std, 
    SKEWNESS(x) AS x_skew, 
    KURTOSIS(x) AS x_kurt, 
    MAX(x) AS x_max,
    PERCENTILE(x, 0.05) AS x_p1,
    PERCENTILE(x, 0.20) AS x_p2,
    PERCENTILE(x, 0.50) AS x_p5,
    PERCENTILE(x, 0.80) AS x_p8,
    PERCENTILE(x, 0.95) AS x_p9,
    AVG(ABS(x)) AS xa_avg,
    STD(ABS(x))AS xa_std, 
    SKEWNESS(ABS(x)) AS xa_skew, 
    KURTOSIS(ABS(x)) AS xa_kurt, 
    MAX(ABS(x)) AS xa_max,
    PERCENTILE(ABS(x), 0.05) AS xa_p1,
    PERCENTILE(ABS(x), 0.20) AS xa_p2,
    PERCENTILE(ABS(x), 0.50) AS xa_p5,
    PERCENTILE(ABS(x), 0.80) AS xa_p8,
    PERCENTILE(ABS(x), 0.95) AS xa_p9
FROM data 
GROUP BY seg
ORDER BY seg
""" )

cols = ["x_avg", "x_std", "x_skew", "x_kurt", "x_max", "x_p1", "x_p2", "x_p5", "x_p8", "x_p9",
        "xa_avg", "xa_std", "xa_skew", "xa_kurt", "xa_max", "xa_p1", "xa_p2", "xa_p5", "xa_p8", "xa_p9"]
assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
df_stat = assembler.transform(df_stat)
for col in cols: df_stat = df_stat.drop(col)


scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
scalerModel = scaler.fit(df_stat)
df_stat = scalerModel.transform(df_stat).drop("stat_t1")

df_stat.write.mode("overwrite").parquet(os.path.join("data", "test.stat.parquet"))


df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_target = spark.sql("""
SELECT seg, y FROM data 
WHERE no = 150000
ORDER BY seg
""" )
df_target.write.mode("overwrite").parquet(os.path.join("data", "train.target.parquet"))

df_test = spark.read.parquet(os.path.join("data", "test.parquet"))
df_test.createOrReplaceTempView("data")
df_target = spark.sql("""
SELECT seg, y FROM data 
WHERE no = 150000
ORDER BY seg
""" )
df_target.write.mode("overwrite").parquet(os.path.join("data", "test.target.parquet"))

df_train = spark.read.parquet(os.path.join("data", "train.vector.parquet"))
df_test = spark.read.parquet(os.path.join("data", "test.vector.parquet"))

for j in range(8):
    cols = ["vx"+str(j+i) for i in range(3)]
    assembler = VectorAssembler(inputCols=cols, outputCol="vx_w"+str(j))
    dct = DCT(inverse=False, inputCol="vx_w"+str(j), outputCol="fr_w"+str(j))
    slicer = VectorSlicer(inputCol="fr_w"+str(j), outputCol="fs_w"+str(j), indices=[i for i in range(12000)])
    scaler = StandardScaler(inputCol="fs_w"+str(j), outputCol="fn_w"+str(j), withStd=True, withMean=False)

    pipeline = Pipeline(stages=[assembler, dct, slicer, scaler])
    pw_model = pipeline.fit(df_train)

    df_train = pw_model.transform(df_train).drop("vx"+str(j)).drop("vx_w"+str(j)).drop("fr_w"+str(j)).drop("fs_w"+str(j))
    df_test = pw_model.transform(df_test).drop("vx"+str(j)).drop("vx_w"+str(j)).drop("fr_w"+str(j)).drop("fs_w"+str(j))

df_train.write.mode("overwrite").parquet(os.path.join("data", "train.win.vector.dct.parquet"))
df_test.write.mode("overwrite").parquet(os.path.join("data", "test.win.vector.dct.parquet"))

df_train.printSchema()
df_test.printSchema()


def bin_win_source_to(source, destination):
    df_w = spark.read.parquet(os.path.join("data", source))
    for j in range(8):
        rdd_w = df_w.rdd.map(lambda row, j=j: (row.seg, row["fn_w" + str(j)], []))
        for i in range(0, 99):
            rdd_w = rdd_w.map(lambda row, i=i: (
            row[0], row[1], row[2] + [float(sum(abs(row[1].toArray()[i * 120:i * 120 + 240])) / 120)]))
        rdd_w = rdd_w.map(lambda row: Row(seg=row[0], f=Vectors.dense(row[2])))
        df_tmp = rdd_w.toDF()
        df_tmp = df_tmp.selectExpr("seg AS seg2", "f AS f" + str(j)).drop("seg").drop("_c0")
        df_w = df_w.join(df_tmp, df_w.seg.cast(IntegerType()) == df_tmp.seg2.cast(IntegerType())).drop("seg2").drop(
            "fn_w" + str(j))
    df_w = df_w.drop("vx8").drop("vx9")
    df_w.write.mode("overwrite").parquet(os.path.join("data", destination))


bin_win_source_to("train.win.vector.dct.parquet", "train.win.vector.fbin.parquet")
bin_win_source_to("test.win.vector.dct.parquet", "test.win.vector.fbin.parquet")


def slice_win_source_to(source, destination):
    df_w = spark.read.parquet(os.path.join("data", source))
    for j in range(8):
        slicer = VectorSlicer(inputCol="f" + str(j), outputCol="f_sl" + str(j), indices=[i for i in range(50, 76)])
        df_w = slicer.transform(df_w).drop("f" + str(j))
    cols = ["f_sl" + str(i) for i in range(8)]
    assembler = VectorAssembler(inputCols=cols, outputCol="f")
    df_w = assembler.transform(df_w)
    df_w.write.mode("overwrite").parquet(os.path.join("data", destination))
    df_w.printSchema()


slice_win_source_to("train.win.vector.fbin.parquet", "train.win.vector.fbin.2.parquet")
slice_win_source_to("test.win.vector.fbin.parquet", "test.win.vector.fbin.2.parquet")

df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_target = spark.sql("""
SELECT d0.seg, d0.y AS y0, d1.y AS y1, d2.y AS y2, d3.y AS y3, 
               d4.y AS y4, d5.y AS y5, d6.y AS y6, d7.y AS y7
FROM        data AS d0 
INNER JOIN  data AS d1 ON d1.no =  60000 AND d1.seg = d0.seg
INNER JOIN  data AS d2 ON d2.no =  75000 AND d2.seg = d0.seg
INNER JOIN  data AS d3 ON d3.no =  90000 AND d3.seg = d0.seg
INNER JOIN  data AS d4 ON d4.no = 105000 AND d4.seg = d0.seg
INNER JOIN  data AS d5 ON d5.no = 120000 AND d5.seg = d0.seg
INNER JOIN  data AS d6 ON d6.no = 135000 AND d6.seg = d0.seg
INNER JOIN  data AS d7 ON d7.no = 150000 AND d7.seg = d0.seg
WHERE d0.no = 45000
ORDER BY d0.seg
""" )

df_target.write.mode("overwrite").parquet(os.path.join("data", "train.win.target.parquet"))

df_target.show()

df_train = spark.read.parquet(os.path.join("data", "train.parquet"))
df_train.createOrReplaceTempView("data")
df_stat = spark.sql("""
SELECT seg,
    INT(no/1000) AS seq,
    AVG(x) AS x_avg,
    PERCENTILE(x, 0.02) AS x_p02,
    PERCENTILE(x, 0.98) AS x_p98,
    PERCENTILE(ABS(x), 0.95) AS xa_p95
FROM data
GROUP BY seg, INT(no/1000)
ORDER BY seg, INT(no/1000)
""" )
df_agg = sc.parallelize([])
rdd_temp = df_stat.rdd.map(lambda row:(row.seg, row.x_avg, row.x_p02, row.x_p98, row.xa_p95))      \
    .map(lambda data: (data[0], ([ data[1] ], [ data[2] ] , [ data[3] ] , [ data[4] ] )))          \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])  )              \
    .map(lambda row: Row(seg=row[0],
                         stx1=Vectors.dense(row[1][0]),
                         stx2=Vectors.dense(row[1][1]),
                         stx3=Vectors.dense(row[1][2]),
                         stx4=Vectors.dense(row[1][3])))
# if df_agg.count() == 0:
df_agg = rdd_temp.toDF(["seg","stx1","stx2","stx3","stx4"])

# df_agg.show()
df_agg = df_agg.select("*").where("seg != 4194")

scaler = StandardScaler(inputCol="stx1", outputCol="stxn1", withStd=True, withMean=False)
scalerModel = scaler.fit(df_agg)
df_agg = scalerModel.transform(df_agg).drop("stx1")

scaler = StandardScaler(inputCol="stx2", outputCol="stxn2", withStd=True, withMean=False)
scalerModel = scaler.fit(df_agg)
df_agg = scalerModel.transform(df_agg).drop("stx2")

scaler = StandardScaler(inputCol="stx3", outputCol="stxn3", withStd=True, withMean=False)
scalerModel = scaler.fit(df_agg)
df_agg = scalerModel.transform(df_agg).drop("stx3")

scaler = StandardScaler(inputCol="stx4", outputCol="stxn4", withStd=True, withMean=False)
scalerModel = scaler.fit(df_agg)
df_agg = scalerModel.transform(df_agg).drop("stx4")

df_agg.write.mode("overwrite").parquet(os.path.join("data", "train.win.stat.parquet"))