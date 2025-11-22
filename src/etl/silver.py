from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, hour, dayofweek, year, unix_timestamp, round

def process_silver():
    spark = SparkSession.builder \
        .appName("SilverLayerProcessing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    bronze_path = "data/bronze"
    silver_path = "data/silver"
    
    print("Processing Silver layer (Cleaning Taxi Data)...")
    
    df = spark.read.parquet(f"{bronze_path}/yellow_tripdata.parquet")
    
    # Data Cleaning & Enrichment
    # 1. Filter invalid trips (negative distance/fare)
    # 2. Add derived columns (date, hour, day_of_week)
    # 3. Calculate Trip Duration (minutes)
    df_clean = df.filter((col("trip_distance") > 0) & (col("fare_amount") > 0)) \
                 .withColumn("trip_date", to_date(col("tpep_pickup_datetime"))) \
                 .withColumn("pickup_year", year(col("tpep_pickup_datetime"))) \
                 .withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
                 .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime"))) \
                 .withColumn("trip_duration_mins", 
                             round((unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60, 2))
    
    # Filter invalid durations (e.g. < 1 min or > 10 hours)
    df_clean = df_clean.filter((col("trip_duration_mins") >= 1) & (col("trip_duration_mins") <= 600))

    # Select relevant columns
    df_final = df_clean.select(
        "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "passenger_count", "trip_distance", "PULocationID", "DOLocationID",
        "payment_type", "fare_amount", "tip_amount", "total_amount",
        "trip_date", "pickup_year", "pickup_hour", "day_of_week", "trip_duration_mins"
    )
    
    df_final.write.mode("overwrite").parquet(f"{silver_path}/trips.parquet")
    
    # Process Zones (Just copy for now, maybe clean if needed)
    print("Processing Zones...")
    df_zones = spark.read.parquet(f"{bronze_path}/zones.parquet")
    df_zones.write.mode("overwrite").parquet(f"{silver_path}/zones.parquet")
    
    print(f"Silver layer processing complete. Cleaned records: {df_final.count()}")
    spark.stop()

if __name__ == "__main__":
    process_silver()
