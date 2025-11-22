from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, desc

def process_gold():
    spark = SparkSession.builder \
        .appName("GoldLayerAggregation") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    silver_path = "data/silver/trips.parquet"
    gold_path = "data/gold"
    
    print("Processing Gold layer (Spark)...")
    
    df = spark.read.parquet(silver_path)
    
    # 1. Daily Trip Stats
    print("Creating Daily Trip Stats...")
    daily_stats = df.groupBy("trip_date") \
        .agg(
            count("*").alias("total_trips"),
            sum("total_amount").alias("total_revenue"),
            avg("trip_distance").alias("avg_distance"),
            avg("fare_amount").alias("avg_fare")
        ) \
        .orderBy("trip_date")
    
    daily_stats.write.mode("overwrite").parquet(f"{gold_path}/daily_stats.parquet")
    
    # 2. Hourly Stats (Rush Hour Analysis)
    print("Creating Hourly Stats...")
    hourly_stats = df.groupBy("pickup_hour") \
        .agg(
            count("*").alias("total_trips"),
            avg("trip_distance").alias("avg_speed_proxy"), # proxy
            avg("total_amount").alias("avg_revenue")
        ) \
        .orderBy("pickup_hour")
        
    hourly_stats.write.mode("overwrite").parquet(f"{gold_path}/hourly_stats.parquet")
    
    # 3. Borough Stats (Geospatial Analysis)
    print("Creating Borough Stats...")
    zones = spark.read.parquet(f"data/silver/zones.parquet")
    
    # Join Trips with Zones (Pickup Location)
    borough_stats = df.join(zones, df.PULocationID == zones.LocationID) \
        .groupBy("Borough") \
        .agg(
            count("*").alias("total_trips"),
            avg("trip_duration_mins").alias("avg_duration_mins"),
            sum("total_amount").alias("total_revenue")
        ) \
        .orderBy(desc("total_trips"))
        
    borough_stats.write.mode("overwrite").parquet(f"{gold_path}/borough_stats.parquet")
    
    print("Gold layer processing complete.")
    spark.stop()

if __name__ == "__main__":
    process_gold()
