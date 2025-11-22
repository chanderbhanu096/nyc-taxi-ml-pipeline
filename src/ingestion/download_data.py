import os
import requests
import time

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def ingest_data():
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    landing_dir = "data/landing"
    
    # Clear existing fake data
    # os.system(f"rm -rf {landing_dir}/*") 
    os.makedirs(landing_dir, exist_ok=True)
    
    # Download All: 2023 and 2024
    years = [2023, 2024]
    months = range(1, 13)
    
    files_to_download = []
    for year in years:
        for month in months:
            # Skip future months if they don't exist yet (e.g. late 2024)
            # For this project we'll try to download all, requests will fail gracefully if not found
            files_to_download.append(f"yellow_tripdata_{year}-{month:02d}.parquet")
    
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        dest_path = os.path.join(landing_dir, filename)
        download_file(url, dest_path)

    # Download Zone Lookup
    zone_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    download_file(zone_url, os.path.join(landing_dir, "taxi_zone_lookup.csv"))

if __name__ == "__main__":
    ingest_data()
