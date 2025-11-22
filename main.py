import os
import sys
import time

def run_pipeline():
    print("=========================================")
    print("   NYC TAXI ANALYTICS PIPELINE")
    print("=========================================")
    
    start_time = time.time()
    
    # Step 1: Data Ingestion
    print("\n[1/5] Running Data Ingestion...")
    exit_code = os.system("python3 src/ingestion/download_data.py")
    if exit_code != 0:
        print("Error in Data Ingestion. Exiting.")
        sys.exit(1)
        
    # Step 2: Bronze Layer
    print("\n[2/5] Running Bronze Layer Ingestion...")
    exit_code = os.system("python3 src/etl/bronze.py")
    if exit_code != 0:
        print("Error in Bronze Layer. Exiting.")
        sys.exit(1)

    # Step 3: Silver Layer
    print("\n[3/5] Running Silver Layer Processing...")
    exit_code = os.system("python3 src/etl/silver.py")
    if exit_code != 0:
        print("Error in Silver Layer. Exiting.")
        sys.exit(1)

    # Step 4: Gold Layer
    print("\n[4/5] Running Gold Layer Aggregation...")
    exit_code = os.system("python3 src/etl/gold.py")
    if exit_code != 0:
        print("Error in Gold Layer. Exiting.")
        sys.exit(1)
        
    # Step 5: Machine Learning
    print("\n[5/5] Running Fare Prediction Model...")
    exit_code = os.system("python3 src/science/predict_sales.py")
    if exit_code != 0:
        print("Error in ML Model. Exiting.")
        sys.exit(1)
        
    end_time = time.time()
    print(f"\nPipeline Completed Successfully in {end_time - start_time:.2f} seconds.")
    print("=========================================")
    print("To view the dashboard, run: streamlit run src/analysis/dashboard.py")

if __name__ == "__main__":
    run_pipeline()
