from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import json
import glob

# Define default arguments with alerting
default_args = {
    'owner': 'Chander Bhanu',
    'depends_on_past': False,
    'email': ['admin@example.com'],  # Configure your email here
    'email_on_failure': True,  # PRODUCTION FEATURE: Email on failure
    'email_on_retry': False,
    'retries': 2,  # Retry twice before failing
    'retry_delay': timedelta(minutes=5),
}

# Get the project root directory
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/Users/main/Desktop/Sparks Project Playground for data Engineer/Project_3")

# Python functions for custom tasks
def check_for_new_files(**context):
    """Check if there are new files to process."""
    incremental_path = f"{PROJECT_ROOT}/data/landing/incremental"
    metadata_file = f"{PROJECT_ROOT}/data/metadata/last_processed.json"
    
    # Load metadata
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        processed = set(metadata.get('processed_files', []))
    else:
        processed = set()
    
    # Find new files
    all_files = glob.glob(f"{incremental_path}/yellow_tripdata_*.parquet")
    new_files = [os.path.basename(f) for f in all_files if os.path.basename(f) not in processed]
    
    if new_files:
        print(f"✅ Found {len(new_files)} new file(s): {new_files}")
        context['ti'].xcom_push(key='new_files', value=new_files)
        return True
    else:
        print("ℹ️  No new files to process")
        return False

with DAG(
    'nyc_taxi_etl_incremental',
    default_args=default_args,
    description='Incremental ETL pipeline for NYC Taxi Data with validation and alerting',
    schedule=timedelta(days=1),  # Check daily for new data
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['etl', 'pyspark', 'nyc_taxi', 'incremental', 'production'],
) as dag:

    # Task 1: Check for New Data
    check_new_data = PythonOperator(
        task_id='check_for_new_data',
        python_callable=check_for_new_files,
    )
    
    # Task 2: Incremental Bronze Layer (with schema validation & idempotency)
    run_bronze_incremental = BashOperator(
        task_id='run_bronze_incremental',
        bash_command=f'cd "{PROJECT_ROOT}" && python3 src/etl/bronze_incremental.py',
    )

    # Task 3: Silver Layer (Cleaning & Enrichment)
    run_silver = BashOperator(
        task_id='run_silver',
        bash_command=f'cd "{PROJECT_ROOT}" && python3 src/etl/silver.py',
    )

    # Task 4: Gold Layer (Aggregation)
    run_gold = BashOperator(
        task_id='run_gold',
        bash_command=f'cd "{PROJECT_ROOT}" && python3 src/etl/gold.py',
    )

    # Define dependencies
    # Check for new data → Run incremental Bronze → Silver → Gold
    check_new_data >> run_bronze_incremental >> run_silver >> run_gold
