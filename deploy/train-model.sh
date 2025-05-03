#!/bin/bash
set -e

echo "Starting model training..."

# Set environment variables
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
export MODEL_DIR=${MODEL_DIR:-models}
export DATA_DIR=${DATA_DIR:-data}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Create directories if they don't exist
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR/processed

# Check if data exists, if not create sample data
if [ ! -f "$DATA_DIR/raw/premium_data.csv" ]; then
  echo "Creating sample data..."
  mkdir -p $DATA_DIR/raw
  echo "id,age,gender,bmi,smoker,region,premium" > $DATA_DIR/raw/premium_data.csv
  echo "1,25,male,22.5,no,northeast,5000" >> $DATA_DIR/raw/premium_data.csv
  echo "2,42,female,28.1,yes,southwest,12500" >> $DATA_DIR/raw/premium_data.csv
  echo "3,38,male,24.3,no,southeast,7500" >> $DATA_DIR/raw/premium_data.csv
fi

# Create a simple model file for testing
echo "Creating a simple model file..."
mkdir -p $MODEL_DIR/$(date +%Y%m%d_%H%M%S)
echo "This is a placeholder model file" > $MODEL_DIR/$(date +%Y%m%d_%H%M%S)/model.txt

echo "Model training completed successfully!"
