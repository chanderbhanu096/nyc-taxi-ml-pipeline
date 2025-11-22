# Quick Start Guide

## 🎯 What This Project Does

This platform:
1. Downloads 24 months of NYC taxi data (~76M trips)
2. Processes it through Bronze → Silver → Gold layers using PySpark
3. Trains 9 ML models to predict fares (best: 95% accurate)
4. Provides an interactive analytics dashboard

---

## ⚡ 5-Minute Setup

```bash
# 1. Clone and setup
git clone https://github.com/chanderbhanu096/nyc-taxi-ml-pipeline.git
cd nyc-taxi-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the pipeline (downloads data + trains models)
# This takes ~30-45 minutes
python3 main.py

# 3. View the dashboard
streamlit run src/analysis/dashboard.py
```

---

## 📁 What Gets Created

After running `main.py`:

```
data/
├── landing/        # Raw downloaded data (15GB)
├── bronze/         # Ingested with metadata
├── silver/         # Cleaned & enriched
└── gold/           # Aggregated for analytics

models/
└── best_fare_model.pkl  # Random Forest model (95% accurate)
```

---

## 🎓 Step-by-Step Tutorial

### Step 1: Download Data Only

```bash
python3 src/ingestion/download_data.py
```

### Step 2: Run Individual ETL Layers

```bash
python3 src/etl/bronze.py   # Raw ingestion
python3 src/etl/silver.py   # Cleaning
python3 src/etl/gold.py     # Aggregation
```

### Step 3: Train ML Models

```bash
.venv/bin/python src/science/predict_sales.py
```

### Step 4: Test Predictions

```bash
python3 src/science/test_model.py
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Dataset** | 76M trips, 2023-2024 |
| **Best Model** | Random Forest |
| **Accuracy** | 94.8% (R² = 0.9475) |
| **MAE** | $1.05 |
| **Features** | 9 (distance, duration, location, etc.) |

---

## 🐛 Troubleshooting

**Issue: Out of Memory**
- Reduce sample size in `predict_sales.py` (line 55): change `0.10` to `0.05` or `0.01`

**Issue: Taking too long**
- Skip data download, use smaller sample
- Models train in ~10-20 min on 5% sample

**Issue: ImportError**
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

---

## 🎯 For Recruiters

This project demonstrates:
- ✅ **Big Data Processing** (PySpark on 76M records)
- ✅ **Data Architecture** (Medallion/Lake house pattern)
- ✅ **ML Engineering** (9 models, cross-validation, 95% accuracy)
- ✅ **Production Patterns** (error handling, modularity, testing)
- ✅ **Visualization** (Streamlit dashboard)

**Time Investment**: ~2 weeks of focused development

---

Need help? Open an issue on GitHub!
