import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("D:/spark-ivy", exist_ok=True)
os.makedirs("D:/spark-tmp", exist_ok=True)

from pyspark.sql import SparkSession

ADD_OPENS = " ".join([f"--add-opens=java.base/{p}=ALL-UNNAMED"
    for p in ["java.lang","java.lang.invoke","java.lang.reflect","java.io",
              "java.net","java.nio","java.util","java.util.concurrent",
              "java.util.concurrent.atomic","sun.nio.ch","sun.nio.cs",
              "sun.security.action","sun.util.calendar"]])

spark = (SparkSession.builder.appName("export").master("local[1]")
    .config("spark.jars.packages",
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1")
    .config("spark.jars.ivy", "D:/spark-ivy")
    .config("spark.local.dir", "D:/spark-tmp")
    .config("spark.driver.memory", "1g")
    .config("spark.driver.extraJavaOptions", ADD_OPENS)
    .config("spark.cassandra.connection.host", "127.0.0.1")
    .config("spark.cassandra.connection.port", "9042")
    .config("spark.cassandra.connection.timeoutMS", "300000")
    .config("spark.cassandra.read.timeoutMS", "300000")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("DOC TU CASSANDRA (khong dem, chi doc)")
print("=" * 60, flush=True)

df = (spark.read.format("org.apache.spark.sql.cassandra")
      .options(keyspace="traffic", table="readings_by_sensor_day")
      .load())

print("DANG GHI RA PARQUET...", flush=True)
df.write.mode("overwrite").parquet("data/readings_cassandra.parquet")

print("XONG! Da luu: data/readings_cassandra.parquet", flush=True)
spark.stop()