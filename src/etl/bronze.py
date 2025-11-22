from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, col
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, TimestampType

import glob
import os
from functools import reduce
from pyspark.sql import DataFrame

def ingest_bronze():
    spark = SparkSession.builder \
        .appName("BronzeLayerIngestion") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    # Re-enable vectorized reader as we are reading file-by-file matching schema
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
    
    landing_path = "data/landing"
    bronze_path = "data/bronze"
    
    print("Ingesting NYC Taxi data into Bronze layer...")
    
    # Get list of all parquet files
    files = sorted(glob.glob(f"{landing_path}/yellow_tripdata_*.parquet"))
    
    if not files:
        print("No data found!")
        return

    # Clear existing bronze data
    if os.path.exists(f"{bronze_path}/yellow_tripdata.parquet"):
        import shutil
        shutil.rmtree(f"{bronze_path}/yellow_tripdata.parquet")

    for file_path in files:
        try:
            # Read single file - Spark infers schema for THIS file only
            df = spark.read.parquet(file_path)
            
            # Standardize columns immediately
            df = df.withColumn("passenger_count", col("passenger_count").cast(DoubleType())) \
                   .withColumn("trip_distance", col("trip_distance").cast(DoubleType())) \
                   .withColumn("RatecodeID", col("RatecodeID").cast(DoubleType())) \
                   .withColumn("VendorID", col("VendorID").cast(LongType())) \
                   .withColumn("PULocationID", col("PULocationID").cast(LongType())) \
                   .withColumn("DOLocationID", col("DOLocationID").cast(LongType())) \
                   .withColumn("payment_type", col("payment_type").cast(LongType())) \
                   .withColumn("fare_amount", col("fare_amount").cast(DoubleType())) \
                   .withColumn("extra", col("extra").cast(DoubleType())) \
                   .withColumn("mta_tax", col("mta_tax").cast(DoubleType())) \
                   .withColumn("tip_amount", col("tip_amount").cast(DoubleType())) \
                   .withColumn("tolls_amount", col("tolls_amount").cast(DoubleType())) \
                   .withColumn("improvement_surcharge", col("improvement_surcharge").cast(DoubleType())) \
                   .withColumn("total_amount", col("total_amount").cast(DoubleType())) \
                   .withColumn("congestion_surcharge", col("congestion_surcharge").cast(DoubleType())) \
                   .withColumn("airport_fee", col("airport_fee").cast(DoubleType()))
            
            # Add metadata
            df = df.withColumn("ingestion_time", current_timestamp()) \
                   .withColumn("source_file", input_file_name())
            
            # Write immediately in append mode
            df.write.mode("append").parquet(f"{bronze_path}/yellow_tripdata.parquet")
            print(f"Processed and written {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    # Ingest Zones
    print("Ingesting Zones...")
    try:
        df_zones = spark.read.option("header", "true").csv(f"{landing_path}/taxi_zone_lookup.csv")
        df_zones.write.mode("overwrite").parquet(f"{bronze_path}/zones.parquet")
        print(f"Ingested Zones.")
    except Exception as e:
        print(f"Error ingesting zones: {e}")
        
    spark.stop()

if __name__ == "__main__":
    ingest_bronze()
