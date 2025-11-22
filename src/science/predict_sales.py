import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import joblib
import os

def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"{name} Results:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R2 Score: {r2:.4f}")
    
    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def predict_fare():
    print("Running Taxi Fare Prediction Model Comparison...")
    
    # Initialize Spark to read large dataset
    spark = SparkSession.builder \
        .appName("FarePredictionSampling") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load Silver Data (Sample for ML)
    try:
        print("Loading data via Spark...")
        # Read the full dataset
        df_spark = spark.read.parquet('data/silver/trips.parquet')
        
        # Sample 10% of data (increased from 5% for better accuracy & generalization)
        print("Sampling 10% of data...")
        df_sample = df_spark.sample(fraction=0.10, seed=42)
        
        # Convert to Pandas
        df = df_sample.toPandas()
        print(f"Loaded {len(df)} records for training.")
        
        spark.stop()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Feature Engineering
    print("Engineering features...")
    
    # Rush hour indicator (7-9am, 5-7pm)
    df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    
    # Weekend indicator
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [1,7] else 0)
    
    # Feature Selection - EXPANDED FEATURES
    features = [
        'trip_distance', 
        'trip_duration_mins',
        'passenger_count',
        'PULocationID',
        'pickup_year', 
        'pickup_hour', 
        'day_of_week',
        'is_rush_hour',
        'is_weekend'
    ]
    target = 'fare_amount'
    
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Models (including XGBoost)
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=42)),
        ("Lasso Regression", Lasso(alpha=1.0, random_state=42)),
        ("ElasticNet", ElasticNet(alpha=1.0, random_state=42)),
        ("Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
        ("Random Forest", RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1))
    ]
    
    results = []
    for name, model in models:
        result = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        
    # Compare and Select Best
    print("\n=========================================")
    print("       MODEL LEADERBOARD (by MAE)       ")
    print("=========================================")
    
    # Sort by MAE (lower is better)
    results.sort(key=lambda x: x['mae'])
    
    for rank, res in enumerate(results, 1):
        print(f"{rank}. {res['name']}: MAE=${res['mae']:.2f}, R2={res['r2']:.4f}")
        
    
    best_model = results[0]
    print(f"\n🏆 Best Model: {best_model['name']}")
    
    # Cross-validate top 3 models to verify generalization
    print("\n" + "="*60)
    print("  GENERALIZATION TEST (5-Fold Cross-Validation)")
    print("="*60)
    
    top_3 = results[:3]
    for res in top_3:
        # Cross-validation on full training set
        cv_scores = cross_val_score(res['model'], X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"{res['name']}:")
        print(f"  Train MAE: ${res['mae']:.2f}")
        print(f"  CV MAE: ${cv_mae:.2f} (±${cv_std:.2f})")
        
        # Check for overfitting
        if cv_mae - res['mae'] > 0.5:
            print(f"  ⚠️  Warning: Possible overfitting (CV worse by ${cv_mae - res['mae']:.2f})")
        else:
            print(f"  ✓ Good generalization!")
        print()
    
    # Save Best Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model['model'], "models/best_fare_model.pkl")
    print("Best model saved to models/best_fare_model.pkl")

if __name__ == "__main__":
    predict_fare()
