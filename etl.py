import os, sys, json
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("D:/spark-ivy", exist_ok=True)
os.makedirs("D:/spark-tmp", exist_ok=True)

from pyspark.sql import SparkSession, functions as F

ADD_OPENS = " ".join([
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
])

spark = (SparkSession.builder.appName("etl-week").master("local[2]")
    .config("spark.jars.packages",
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.1")
    .config("spark.jars.ivy", "D:/spark-ivy")
    .config("spark.local.dir", "D:/spark-tmp")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC " + ADD_OPENS)
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.sql.files.maxPartitionBytes", "32m")
    .config("spark.cassandra.connection.host", "127.0.0.1")
    .config("spark.cassandra.connection.port", "9042")
    # ==== Hạ tải để tránh timeout ====
    .config("spark.cassandra.output.concurrent.writes", "1")
    .config("spark.cassandra.output.batch.size.rows", "5")       # 20 -> 5
    .config("spark.cassandra.output.batch.size.bytes", "512")    # 1024 -> 512
    .config("spark.cassandra.output.batch.grouping.buffer.size", "100")
    .config("spark.cassandra.output.throughput_mb_per_sec", "1") # giới hạn 1 MB/s
    .config("spark.cassandra.query.retry.count", "20")
    .config("spark.cassandra.connection.timeoutMS", "60000")
    .config("spark.cassandra.read.timeoutMS", "60000")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

full = spark.read.parquet("data/pems_bay_long_ms.parquet")
print("Tổng số dòng trong Parquet:", full.count(), flush=True)

df = (full.withColumn("speed", F.col("speed").cast("double"))
          .withColumn("day", F.to_date("ts")))
days = [f"2017-01-{d:02d}" for d in range(1, 32)]   

for d in days:
    print(f"=== Đang ghi ngày {d} ===", flush=True)
    batch = df.filter(F.to_date("ts") == F.lit(d).cast("date"))
    n = batch.count()
    print(f"  -> có {n} dòng", flush=True)
    (batch.select("sensor_id", "day", "ts", "speed")
       .write.format("org.apache.spark.sql.cassandra")
       .options(table="readings_by_sensor_day", keyspace="traffic")
       .mode("append").save())
    print(f"=== Xong ngày {d} ===", flush=True)

print("TẤT CẢ XONG", flush=True)
spark.stop()