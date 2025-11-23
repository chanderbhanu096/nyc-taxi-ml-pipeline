# Incremental ETL Demo Guide

## 🎯 Goal
Demonstrate the production-grade incremental ETL pipeline with schema validation, idempotency, and late data handling.

## 📋 Pre-requisites
- ✅ Baseline load complete (2023-2024 data processed)
- ✅ Airflow running (`./start_airflow.sh`)
- ✅ Metadata tracking initialized

---

## 🚀 Demo Flow

### Step 1: Verify Baseline Status
```bash
cat data/metadata/last_processed.json
```
Expected output:
```json
{
  "baseline_load_complete": true,
  "baseline_end_date": "2024-12-31",
  ...
}
```

### Step 2: Simulate New Monthly Data Arrival

Pick one existing file from `data/landing/` and copy it as "2025-01" data:

```bash
# Copy an existing month as simulated "2025-01" data
cp data/landing/yellow_tripdata_2023-12.parquet data/landing/incremental/yellow_tripdata_2025-01.parquet
```

### Step 3: Trigger Airflow DAG

**Option A: Automatic (wait for schedule)**
- DAG runs daily at midnight
- Will detect new file automatically

**Option B: Manual Trigger (recommended for demo)**
1. Open Airflow UI: http://localhost:8080
2. Find DAG: `nyc_taxi_etl_incremental`
3. Click "Trigger DAG" ▶️

### Step 4: Monitor Execution

Watch the DAG execution:
1. **check_for_new_data** → Should detect `yellow_tripdata_2025-01.parquet`
2. **run_bronze_incremental** → Validates schema, checks idempotency, ingests data
3. **run_silver** → Cleans new records
4. **run_gold** → Updates aggregations

### Step 5: Verify Results

**A. Check Metadata:**
```bash
cat data/metadata/last_processed.json
```
Should now show:
```json
{
  "last_incremental_month": "2025-01",
  "processed_files": ["yellow_tripdata_2025-01.parquet"],
  ...
}
```

**B. Check Dashboard:**
```bash
# If not running: streamlit run src/analysis/dashboard.py
```
Navigate to "Pipeline Status" page → Should show:
- Last Incremental: `2025-01`
- Processed files: 1

---

## 🧪 Testing Production Features

### Test 1: Idempotency
**What:** Re-run the same file → Should skip

**Steps:**
1. Trigger DAG again (without adding new files)
2. Check logs for `run_bronze_incremental`
3. Expected: "Already processed (skipping)"

### Test 2: Schema Validation
**What:** Submit a corrupted file → Should fail gracefully

**Steps:**
```bash
# Create a dummy file with wrong schema
echo "invalid data" > data/landing/incremental/yellow_tripdata_2025-02.parquet
```
1. Trigger DAG
2. Expected: Task fails with "Schema validation FAILED"
3. Clean up: `rm data/landing/incremental/yellow_tripdata_2025-02.parquet`

### Test 3: Late Data Handling
**What:** Submit "January" after "March" → Both should process

**Steps:**
```bash
# Add March first
cp data/landing/yellow_tripdata_2024-01.parquet data/landing/incremental/yellow_tripdata_2025-03.parquet

# Trigger DAG → processes 2025-03

# Then add January (late arrival)
cp data/landing/yellow_tripdata_2024-02.parquet data/landing/incremental/yellow_tripdata_2025-01.parquet

# Trigger DAG → processes 2025-01
```

Both files should be tracked in `processed_files` with separate `arrival_timestamps`.

### Test 4: Email Alerting
**What:** DAG failure sends email (if SMTP configured)

**Steps:**
1. Kill PySpark mid-run (simulate crash)
2. DAG fails
3. Check email (if configured in `default_args['email']`)

---

## 🎥 Portfolio Presentation

**Key talking points:**
1. "I implemented incremental ETL to process only new monthly data, reducing runtime from hours to minutes."
2. "Schema validation prevents corrupted data from entering the pipeline."
3. "Idempotency ensures safe re-runs without duplicates."
4. "Late data handling tracks arrival vs data dates for audit trails."
5. "Email alerting notifies me immediately on failures."

**Visual proof:**
- ✅ Show Airflow DAG graph (4 task flow)
- ✅ Show metadata file evolution
- ✅ Show dashboard "Incremental Pipeline Status" section
- ✅ Show logs with schema validation messages

---

## 📸 Screenshots to Capture

1. Airflow UI - DAG Graph (shows 4-task flow)
2. Airflow UI - Grid View (successful run)
3. Dashboard - Pipeline Status page (incremental metadata)
4. Terminal - `bronze_incremental.py` output (schema validation logs)
5. `last_processed.json` file (metadata evolution)

---

## 🔄 Reset Demo (Start Over)

```bash
# Clear incremental folder
rm -f data/landing/incremental/*.parquet

# Reset metadata
cat > data/metadata/last_processed.json << 'EOF'
{
  "baseline_load_complete": true,
  "baseline_end_date": "2024-12-31",
  "last_incremental_month": null,
  "processed_files": [],
  "arrival_timestamps": {},
  "last_dag_run": null
}
EOF
```

---

**Ready to impress! 🚀**
