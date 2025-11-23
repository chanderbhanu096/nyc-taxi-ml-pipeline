"""
Incremental Bronze Layer Ingestion for NYC Taxi Data.
Processes only new monthly files from data/landing/incremental/.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, col
from pyspark.sql.types import LongType, DoubleType
import glob
import os
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.schema_validator import validate_schema

METADATA_FILE = "data/metadata/last_processed.json"

def load_metadata():
    """Load processing metadata."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {
        "baseline_load_complete": False,
        "last_incremental_month": None,
        "processed_files": [],
        "arrival_timestamps": {}
    }

def save_metadata(metadata):
    """Save processing metadata."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def ingest_incremental():
    """
    Ingest only new monthly files from data/landing/incremental/.
    Includes schema validation and idempotency checks.
    """
    spark = SparkSession.builder \
        .appName("BronzeLayerIncremental") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
    
    incremental_path = "data/landing/incremental"
    bronze_path = "data/bronze"
    
    print("="*60)
    print("🔄 INCREMENTAL BRONZE LAYER INGESTION")
    print("="*60)
    
    # Load metadata
    metadata = load_metadata()
    processed_files = set(metadata.get("processed_files", []))
    
    # Find new files
    all_files = sorted(glob.glob(f"{incremental_path}/yellow_tripdata_*.parquet"))
    new_files = [f for f in all_files if os.path.basename(f) not in processed_files]
    
    if not new_files:
        print("✅ No new files to process. All incremental data is up to date.")
        spark.stop()
        return
    
    print(f"📁 Found {len(new_files)} new file(s) to process:")
    for f in new_files:
        print(f"   - {os.path.basename(f)}")
    print()
    
    # Process each new file
    for file_path in new_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Step 1: Schema Validation
        print("  ➜ Validating schema...")
        is_valid, message = validate_schema(file_path)
        if not is_valid:
            print(f"  ❌ Schema validation FAILED: {message}")
            print(f"  ⏭️  Skipping {filename}")
            continue
        print(f"  ✅ {message}")
        
        # Step 2: Idempotency Check
        if filename in processed_files:
            print(f"  ⏭️  Already processed (skipping)")
            continue
        
        # Step 3: Read and Transform
        try:
            print("  ➜ Reading data...")
            df = spark.read.parquet(file_path)
            
            # Standardize columns
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
            
            # Write in APPEND mode
            print("  ➜ Writing to Bronze layer (append mode)...")
            df.write.mode("append").parquet(f"{bronze_path}/yellow_tripdata.parquet")
            print(f"  ✅ Successfully ingested {filename}")
            
            # Update metadata
            processed_files.add(filename)
            metadata["processed_files"] = list(processed_files)
            metadata["arrival_timestamps"][filename] = datetime.now().isoformat()
            
            # Extract month from filename (e.g., "2025-01")
            month = filename.split("_")[-1].replace(".parquet", "")
            metadata["last_incremental_month"] = month
            metadata["last_dag_run"] = datetime.now().isoformat()
            
            save_metadata(metadata)
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            continue
    
    print("="*60)
    print(f"✅ Incremental ingestion complete. Processed {len(new_files)} file(s).")
    print("="*60)
    spark.stop()

if __name__ == "__main__":
    ingest_incremental()
