"""
Production-grade schema validation utility for NYC Taxi data.
Implements forward-compatible validation with type flexibility.
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, LongType, DoubleType, TimestampType,
    IntegerType, StringType, TimestampNTZType
)
import sys

# Required baseline columns (minimum schema)
REQUIRED_COLUMNS = {
    "VendorID": [LongType, IntegerType],  # Accept both
    "tpep_pickup_datetime": [TimestampType, TimestampNTZType],
    "tpep_dropoff_datetime": [TimestampType, TimestampNTZType],
    "passenger_count": [DoubleType, LongType, IntegerType],
    "trip_distance": [DoubleType],
    "RatecodeID": [DoubleType, LongType, IntegerType],
    "store_and_fwd_flag": [DoubleType, StringType],  # Can be string or numeric
    "PULocationID": [LongType, IntegerType],
    "DOLocationID": [LongType, IntegerType],
    "payment_type": [LongType, IntegerType],
    "fare_amount": [DoubleType],
    "extra": [DoubleType],
    "mta_tax": [DoubleType],
    "tip_amount": [DoubleType],
    "tolls_amount": [DoubleType],
    "improvement_surcharge": [DoubleType],
    "total_amount": [DoubleType],
    "congestion_surcharge": [DoubleType],
}

def validate_schema(file_path: str) -> tuple[bool, str]:
    """
    Forward-compatible schema validation.
    
    ✅ Allows NEW columns (e.g., airport_fee)
    ✅ Accepts compatible types (Int ≈ Long)
    ❌ Fails if REQUIRED columns are missing
    
    Args:
        file_path: Path to the parquet file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        spark = SparkSession.builder \
            .appName("SchemaValidator") \
            .getOrCreate()
        
        # Read the file
        df = spark.read.parquet(file_path)
        actual_schema = df.schema
        actual_cols = {field.name: type(field.dataType) for field in actual_schema.fields}
        
        # ==========================================
        # 1. CHECK FOR MISSING REQUIRED COLUMNS
        # ==========================================
        missing_cols = set(REQUIRED_COLUMNS.keys()) - set(actual_cols.keys())
        if missing_cols:
            return False, f"❌ Missing required columns: {missing_cols}"
        
        # ==========================================
        # 2. CHECK TYPE COMPATIBILITY
        # ==========================================
        type_errors = []
        for col_name, allowed_types in REQUIRED_COLUMNS.items():
            if col_name in actual_cols:
                actual_type = actual_cols[col_name]
                # Check if actual type is in the list of allowed types
                if not any(actual_type == allowed_type for allowed_type in allowed_types):
                    type_errors.append(
                        f"{col_name}: got {actual_type.__name__}, "
                        f"expected one of {[t.__name__ for t in allowed_types]}"
                    )
        
        if type_errors:
            return False, f"❌ Type mismatches:\n  " + "\n  ".join(type_errors)
        
        # ==========================================
        # 3. DETECT NEW COLUMNS (WARNING ONLY)
        # ==========================================
        extra_cols = set(actual_cols.keys()) - set(REQUIRED_COLUMNS.keys())
        if extra_cols:
            print(f"⚠️  SCHEMA EVOLUTION DETECTED: New columns found: {extra_cols}")
            print(f"ℹ️  Forward compatibility: Processing will continue.")
        
        # ==========================================
        # SUCCESS
        # ==========================================
        print(f"✅ Schema validation PASSED for {file_path}")
        if extra_cols:
            print(f"   📝 Note: {len(extra_cols)} new column(s) detected and accepted")
        return True, "Schema is valid (forward-compatible)"
        
    except Exception as e:
        return False, f"❌ Error reading file: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python schema_validator.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    is_valid, message = validate_schema(file_path)
    
    print(message)
    sys.exit(0 if is_valid else 1)
