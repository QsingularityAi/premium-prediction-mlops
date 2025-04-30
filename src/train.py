#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Script for Premium Prediction
This script trains the segmented model pipeline and logs results to MLflow.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'train.log'))
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train premium prediction models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data-path', type=str, 
                      help='Override data path from config')
    parser.add_argument('--model-dir', type=str,
                      help='Override model directory from config')
    parser.add_argument('--experiment-name', type=str,
                      help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str,
                      help='MLflow tracking URI')
    parser.add_argument('--log-artifacts', action='store_true', default=True,
                      help='Whether to log artifacts to MLflow')
    parser.add_argument('--register-model', action='store_true', default=True,
                      help='Whether to register the model in MLflow Model Registry')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def setup_mlflow(config: Dict[str, Any], args: argparse.Namespace) -> str:
    """
    Set up MLflow tracking.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        MLflow experiment ID
    """
    # Set tracking URI
    tracking_uri = args.tracking_uri or config['mlflow'].get('tracking_uri', 'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Set experiment name
    experiment_name = args.experiment_name or config['mlflow'].get('experiment_name', 'premium_prediction')
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=config['mlflow'].get('artifact_location', 'artifacts')
        )
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    return experiment_id

def save_model_metadata(
    models_dir: Path, 
    premium_thresholds: Dict[str, float], 
    combined_metrics: Dict[str, float],
    segment_metrics: Dict[str, Dict[str, Any]],
    version: str
) -> None:
    """
    Save model metadata to JSON files.
    
    Args:
        models_dir: Directory to save metadata
        premium_thresholds: Dictionary of premium thresholds
        combined_metrics: Dictionary of combined model metrics
        segment_metrics: Dictionary of segment-specific metrics
        version: Model version
    """
    version_dir = models_dir / version
    version_dir.mkdir(exist_ok=True, parents=True)
    
    # Save premium thresholds
    with open(version_dir / 'premium_thresholds.json', 'w') as f:
        json.dump(premium_thresholds, f, indent=2)
    
    # Save metrics
    metrics = {
        'combined': combined_metrics,
        'segments': segment_metrics,
        'timestamp': datetime.now().isoformat(),
        'version': version
    }
    
    with open(version_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Model metadata saved to {version_dir}")

def main():
    """Main function to run the training pipeline."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override configurations with command line arguments if provided
        if args.data_path:
            config['data']['data_path'] = args.data_path
        if args.model_dir:
            config['paths']['model_dir'] = args.model_dir
        
        # Setup MLflow
        experiment_id = setup_mlflow(config, args)
            
        # Initialize data processor
        data_processor = DataProcessor(config)
        
        # Process data
        logger.info("Loading and processing data...")
        df = data_processor.load_data()
        df = data_processor.preprocess_data(df)
        df = data_processor.perform_feature_engineering(df)
        
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(df)
        
        # Get feature types
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        
        # Create preprocessor
        preprocessor = data_processor.create_preprocessing_pipeline(numerical_features, categorical_features)
        
        # Create segments
        premium_thresholds = data_processor.create_segments(y_train)
        segment_data_train = data_processor.split_into_segments(X_train, y_train, premium_thresholds)
        test_masks = data_processor.get_segment_masks_test(y_test, premium_thresholds)
        
        # Initialize model trainer
        model_trainer = ModelTrainer(config)
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started: {run_id}")
            
            # Log parameters
            mlflow.log_params({
                "target_column": config['data']['target_column'],
                "test_size": config['data']['test_size'],
                "random_state": config['training']['random_state'],
                "n_iter_search": config['training']['n_iter_search'],
                "feature_selection_threshold": config['training']['feature_selection_threshold'],
                "num_segments": len(segment_data_train),
                "num_features": X_train.shape[1],
                "num_samples": X_train.shape[0]
            })
            
            # Define model configs
            segment_model_configs = model_trainer.define_model_configs(segment_data_train)
            
            # Train models
            logger.info("Training segmented models...")
            trained_models = model_trainer.train_segment_models(segment_data_train, preprocessor, segment_model_configs)
            
            # Evaluate models
            logger.info("Evaluating models...")
            y_pred, combined_metrics, segment_metrics = model_trainer.evaluate_models(
                trained_models, X_test, y_test, test_masks
            )
            
            # Log metrics
            mlflow.log_metrics({
                "overall_rmse": combined_metrics['rmse'],
                "overall_mae": combined_metrics['mae'],
                "overall_r2": combined_metrics['r2']
            })
            
            # Log segment-specific metrics
            for segment, metrics in segment_metrics.items():
                if metrics['count'] > 0 and metrics['r2'] is not None:
                    mlflow.log_metrics({
                        f"{segment}_rmse": metrics['rmse'],
                        f"{segment}_mae": metrics['mae'],
                        f"{segment}_r2": metrics['r2']
                    })
            
            # Create model version using timestamp
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = Path(config['paths']['model_dir'])
            version_dir = models_dir / model_version
            version_dir.mkdir(exist_ok=True, parents=True)
            
            # Save models, preprocessor, and thresholds
            logger.info(f"Saving models to {version_dir}...")
            
            # Save preprocessor
            joblib_path = version_dir / "preprocessor.joblib"
            import joblib
            joblib.dump(preprocessor, joblib_path)
            
            # Save segment models
            for segment, model in trained_models.items():
                if model is not None:
                    joblib.dump(model, version_dir / f"{segment}_model.joblib")
            
            # Save thresholds and metrics
            save_model_metadata(models_dir, premium_thresholds, combined_metrics, segment_metrics, model_version)
            
            # Generate and save visualization plots if log_artifacts is enabled
            if args.log_artifacts:
                logger.info("Generating visualization plots...")
                
                # Create artifact directory
                artifact_dir = Path("artifacts") / model_version
                artifact_dir.mkdir(exist_ok=True, parents=True)
                
                # Actual vs Predicted plot
                plt.figure(figsize=(8, 8))
                plt.scatter(y_test, y_pred, alpha=0.3, s=10, label=f'Overall R²: {combined_metrics["r2"]:.3f}')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect')
                plt.xlabel('Actual Premium')
                plt.ylabel('Predicted Premium')
                plt.title('Actual vs Predicted Premiums (Segmented Model)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(artifact_dir / 'actual_vs_predicted.png')
                plt.close()
                
                # Error Analysis
                errors = y_test - y_pred
                abs_errors = np.abs(errors)
                
                # Error distribution
                plt.figure(figsize=(10, 5))
                sns.histplot(errors, bins=50, kde=True)
                plt.xlabel('Prediction Error (Actual - Predicted)')
                plt.title('Distribution of Prediction Errors')
                plt.grid(True, alpha=0.3)
                plt.savefig(artifact_dir / 'error_distribution.png')
                plt.close()
                
                # Log artifacts to MLflow
                mlflow.log_artifacts(str(artifact_dir))
                
                # Analyze feature importance if possible
                try:
                    feature_importance_data = model_trainer.analyze_feature_importance(
                        trained_models, segment_data_train, str(artifact_dir)
                    )
                    
                    # Log feature importance plots
                    for segment, importance_file in feature_importance_data.items():
                        if importance_file:
                            mlflow.log_artifact(importance_file)
                except Exception as e:
                    logger.warning(f"Error generating feature importance plots: {e}")
            
            # Register the model if requested
            if args.register_model:
                try:
                    for segment, model in trained_models.items():
                        if model is not None:
                            mlflow.sklearn.log_model(
                                model,
                                artifact_path=f"models/{segment}",
                                registered_model_name=f"premium_prediction_{segment}"
                            )
                    logger.info("Models registered in MLflow Model Registry")
                except Exception as e:
                    logger.warning(f"Error registering models: {e}")
            
            logger.info(f"Training completed successfully. Models saved to {version_dir}")
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Run ID: {run_id}")
            
            # Print summary
            print("\n========== TRAINING SUMMARY ==========")
            print(f"Model Version: {model_version}")
            print(f"Overall R²:   {combined_metrics['r2']:.4f}")
            print(f"Overall RMSE: {combined_metrics['rmse']:.2f}")
            print(f"Overall MAE:  {combined_metrics['mae']:.2f}")
            print("\nSegment Performance:")
            for segment, metrics in segment_metrics.items():
                if metrics['count'] > 0 and metrics['r2'] is not None:
                    print(f"  {segment.capitalize():<10}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}")
            print("\nModels saved to:", version_dir)
            print("MLflow tracking URI:", mlflow.get_tracking_uri())
            print("Experiment ID:", experiment_id)
            print("Run ID:", run_id)
            print("======================================\n")
            
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

