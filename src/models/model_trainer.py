import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib

# Machine learning imports
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Parameter distributions
from scipy.stats import randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class responsible for training, evaluating, and saving models.
    Supports segmented training of LGBM and XGBoost models.
    """

    def __init__(self, config: Dict):
        """
        Initialize the ModelTrainer with configuration parameters.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.random_state = config["training"]["random_state"]
        self.n_iter_search = config["training"]["n_iter_search"]
        self.cv_folds_tuning = config["training"]["cv_folds_tuning"]
        self.cv_folds_eval = config["training"]["cv_folds_eval"]
        self.feature_selection_threshold = config["training"][
            "feature_selection_threshold"
        ]
        self.model_dir = Path(config["paths"]["model_dir"])
        self.trained_models = {}

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def define_model_configs(self, segment_data_train: Dict) -> Dict:
        """
        Define model configurations for each segment.

        Args:
            segment_data_train: Dictionary mapping segment names to (X, y) tuples

        Returns:
            Dictionary of model configurations for each segment
        """
        # Parameter distributions for tuning
        lgbm_param_dist = {
            "model__n_estimators": randint(100, 800),
            "model__learning_rate": uniform(0.005, 0.05),
            "model__num_leaves": randint(15, 60),
            "model__max_depth": randint(3, 10),
            "model__min_child_samples": randint(10, 50),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": uniform(0.0, 1.0),
            "model__reg_lambda": uniform(0.0, 1.0),
        }

        xgb_param_dist = {
            "model__n_estimators": randint(100, 800),
            "model__learning_rate": uniform(0.005, 0.05),
            "model__max_depth": randint(3, 9),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__gamma": uniform(0, 0.5),
            "model__reg_alpha": uniform(0.0, 1.0),
            "model__reg_lambda": uniform(0.0, 1.0),
        }

        # Initialize segment model configs
        segment_model_configs = {}

        # Iterate through segments in the config
        for segment, segment_config in self.config["segments"].items():
            # Get model type and log transform setting from config
            model_type = segment_config.get("model_type", "lgbm")
            use_log_transform = segment_config.get("use_log_transform", False)

            # Create base model based on config
            if model_type.lower() == "lgbm":
                # Start with default parameters
                model_params = {
                    "objective": "regression",
                    "random_state": self.random_state,
                    "n_jobs": -1,
                    "verbose": -1,
                    "feature_name": "auto",
                }

                # Add hyperparameters from config if available
                if "hyperparams" in segment_config:
                    model_params.update(segment_config["hyperparams"])

                model_base = lgb.LGBMRegressor(**model_params)
                selector = SelectFromModel(
                    model_base, threshold=self.feature_selection_threshold, prefit=False
                )
                params = lgbm_param_dist

            elif model_type.lower() == "xgb":
                # Start with default parameters
                model_params = {
                    "objective": "reg:squarederror",
                    "random_state": self.random_state,
                    "n_jobs": -1,
                }

                # Add hyperparameters from config if available
                if "hyperparams" in segment_config:
                    model_params.update(segment_config["hyperparams"])

                model_base = xgb.XGBRegressor(**model_params)
                selector = SelectFromModel(
                    model_base, threshold=self.feature_selection_threshold, prefit=False
                )
                params = xgb_param_dist
            else:
                logger.warning(
                    f"Unknown model type '{model_type}' for segment '{segment}'. Using default LGBM."
                )
                model_base = lgb.LGBMRegressor(
                    objective="regression",
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    feature_name="auto",
                )
                selector = SelectFromModel(
                    model_base, threshold=self.feature_selection_threshold, prefit=False
                )
                params = lgbm_param_dist

            # Add to config dictionary
            segment_model_configs[segment] = {
                "model": model_base,
                "selector": selector,
                "params": params,
                "use_log_transform": use_log_transform,
            }

            logger.info(
                f"Configured {segment} segment with {model_type} model, log_transform={use_log_transform}"
            )
            if "hyperparams" in segment_config:
                logger.info(f"  Hyperparameters: {segment_config['hyperparams']}")

        # Check if we have all required segments
        expected_segments = ["very_low", "low", "medium", "high", "very_high"]
        for segment in expected_segments:
            if segment not in segment_model_configs:
                logger.warning(
                    f"Missing configuration for '{segment}' segment. Using default LGBM config."
                )

                # Create default configuration
                model_base = lgb.LGBMRegressor(
                    objective="regression",
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    feature_name="auto",
                )
                selector = SelectFromModel(
                    model_base, threshold=self.feature_selection_threshold, prefit=False
                )

                segment_model_configs[segment] = {
                    "model": model_base,
                    "selector": selector,
                    "params": lgbm_param_dist,
                    "use_log_transform": False,
                }

        return segment_model_configs

    def train_segment_models(
        self,
        segment_data_train: Dict,
        preprocessor: Any,
        segment_model_configs: Optional[Dict] = None,
    ) -> Dict:
        """
        Train models for each segment.

        Args:
            segment_data_train: Dictionary mapping segment names to (X, y) tuples
            preprocessor: Preprocessing pipeline
            segment_model_configs: Optional model configurations (if None, will be created)

        Returns:
            Dictionary of trained models for each segment
        """
        logger.info("Training and tuning segment-specific models...")

        if segment_model_configs is None:
            segment_model_configs = self.define_model_configs(segment_data_train)

        trained_models = {}
        metadata = {}

        for segment, config in segment_model_configs.items():
            X_seg, y_seg = segment_data_train[segment]

            # Skip if too few samples for cross-validation
            if len(X_seg) < self.cv_folds_tuning * 2:
                logger.info(
                    f"Skipping segment '{segment}' due to insufficient data ({len(X_seg)} samples) for CV={self.cv_folds_tuning}."
                )
                trained_models[segment] = None
                continue

            logger.info(f"--- Tuning {segment.capitalize()} segment model ---")

            # Create pipeline
            pipeline_to_tune = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("selector", config["selector"]),
                    ("model", config["model"]),
                ]
            )

            # Apply log transformation if configured
            if config["use_log_transform"]:
                final_estimator = TransformedTargetRegressor(
                    regressor=pipeline_to_tune, func=np.log1p, inverse_func=np.expm1
                )
                search_param_dist = {
                    f"regressor__{k}": v for k, v in config["params"].items()
                }
            else:
                final_estimator = pipeline_to_tune
                search_param_dist = config["params"]

            # Check if hyperparameters are fixed in config
            segment_config = self.config["segments"].get(segment, {})
            hyperparams = segment_config.get("hyperparams", {})

            # If we have specific hyperparameters in config, use them directly instead of RandomizedSearchCV
            if hyperparams and all(
                key in hyperparams
                for key in ["num_leaves", "learning_rate", "n_estimators"]
            ):
                logger.info(
                    f"Using fixed hyperparameters for {segment} segment from config."
                )

                # Train the model directly with specified hyperparameters
                try:
                    final_estimator.fit(X_seg, y_seg)
                    trained_models[segment] = final_estimator

                    # Store metadata about the model
                    metadata[segment] = {
                        "training_samples": int(len(X_seg)),
                        "model_type": config["model"].__class__.__name__,
                        "log_transform": bool(config["use_log_transform"]),
                        "fixed_hyperparams": hyperparams,
                        "timestamp": datetime.now().isoformat(),
                    }

                    logger.info(f"Trained {segment} model with fixed hyperparameters.")
                except Exception as e:
                    logger.error(f"ERROR during training for segment {segment}: {e}")
                    trained_models[segment] = None
            else:
                # Perform randomized search for hyperparameter tuning
                random_search = RandomizedSearchCV(
                    estimator=final_estimator,
                    param_distributions=search_param_dist,
                    n_iter=self.n_iter_search,
                    cv=self.cv_folds_tuning,
                    scoring="neg_root_mean_squared_error",
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=1,
                    error_score="raise",
                )

                try:
                    random_search.fit(X_seg, y_seg)
                    best_model = random_search.best_estimator_
                    best_score = random_search.best_score_
                    best_params = random_search.best_params_

                    trained_models[segment] = best_model

                    # Store metadata about the model
                    metadata[segment] = {
                        "best_score": float(best_score),
                        "best_params": {
                            k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in best_params.items()
                        },
                        "training_samples": int(len(X_seg)),
                        "model_type": config["model"].__class__.__name__,
                        "log_transform": bool(config["use_log_transform"]),
                        "timestamp": datetime.now().isoformat(),
                    }

                    logger.info(
                        f"Best score (neg RMSE) for {segment}: {best_score:.4f}"
                    )
                except Exception as e:
                    logger.error(f"ERROR during tuning for segment {segment}: {e}")
                    trained_models[segment] = None

        # Save metadata
        # self.save_model_metadata(metadata) # Temporarily commented out for debugging NameError

        return trained_models

    def evaluate_models(
        self,
        trained_models: Dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_masks: Dict,
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Evaluate trained models on test data.

        Args:
            trained_models: Dictionary of trained models for each segment
            X_test: Test features
            y_test: Test target
            test_masks: Dictionary of test data masks for each segment

        Returns:
            Tuple containing (combined predictions, combined metrics, segment metrics)
        """
        logger.info("Making predictions on the test set...")

        # Initialize prediction array
        y_pred_combined = np.zeros_like(y_test, dtype=float)
        prediction_successful = True

        # Make predictions for each segment
        for segment, mask in test_masks.items():
            X_seg_test = X_test[mask]
            model = trained_models.get(segment)

            if model is not None and len(X_seg_test) > 0:
                try:
                    y_pred_segment = model.predict(X_seg_test)
                    y_pred_segment = np.maximum(
                        0, y_pred_segment
                    )  # Ensure predictions are non-negative
                    y_pred_combined[mask] = y_pred_segment
                except Exception as e:
                    logger.error(f"ERROR predicting for segment {segment}: {e}")
                    prediction_successful = False
            elif len(X_seg_test) > 0:
                logger.warning(
                    f"WARNING: Model for segment '{segment}' not trained. Predictions zero."
                )
                prediction_successful = False

        # Calculate evaluation metrics
        if prediction_successful:
            logger.info("--- Combined Segmented Model Performance ---")

            # Calculate overall metrics
            combined_mae = mean_absolute_error(y_test, y_pred_combined)
            combined_rmse = np.sqrt(mean_squared_error(y_test, y_pred_combined))
            combined_r2 = r2_score(y_test, y_pred_combined)

            combined_metrics = {
                "mae": combined_mae,
                "rmse": combined_rmse,
                "r2": combined_r2,
            }

            logger.info(f"Overall MAE:  {combined_mae:.2f}")
            logger.info(f"Overall RMSE: {combined_rmse:.2f}")
            logger.info(f"Overall R²:   {combined_r2:.4f}")

            # Calculate segment-specific metrics
            logger.info("--- Performance within Test Segments ---")
            segment_metrics = {}

            for segment, mask in test_masks.items():
                y_seg_test = y_test[mask]
                y_seg_pred = y_pred_combined[mask]
                count = len(y_seg_test)

                if count > 0:
                    mae = mean_absolute_error(y_seg_test, y_seg_pred)
                    rmse = np.sqrt(mean_squared_error(y_seg_test, y_seg_pred))
                    r2 = (
                        r2_score(y_seg_test, y_seg_pred)
                        if np.var(y_seg_test) > 1e-9
                        else np.nan
                    )

                    segment_metrics[segment] = {
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "r2": float(r2) if not np.isnan(r2) else None,
                        "count": int(count),
                    }

                    r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
                    logger.info(
                        f"{segment.capitalize():<10} (N={count:<5}): MAE={mae:<7.2f} RMSE={rmse:<7.2f} R²={r2_str:<7}"
                    )
                else:
                    segment_metrics[segment] = {
                        "mae": None,
                        "rmse": None,
                        "r2": None,
                        "count": 0,
                    }
                    logger.info(
                        f"{segment.capitalize():<10} (N={count:<5}): No test samples"
                    )
        else:
            logger.warning("Evaluation skipped due to prediction errors.")
            combined_metrics = {"mae": None, "rmse": None, "r2": None}
            segment_metrics = {
                segment: {"mae": None, "rmse": None, "r2": None, "count": 0}
                for segment in test_masks
            }

        return y_pred_combined, combined_metrics, segment_metrics

    def analyze_feature_importance(
        self,
        trained_models: Dict,
        segment_data_train: Dict,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Analyze feature importance for each segment model.

        Args:
            trained_models: Dictionary of trained models for each segment
            segment_data_train: Dictionary mapping segment names to (X, y) tuples
            output_dir: Directory to save feature importance plots

        Returns:
            Dictionary mapping segment names to feature importance dataframes.
        """
        # Placeholder for implementation
        logger.warning("Feature importance analysis not fully implemented.")
        feature_importances = {}
        # Add logic here to extract and plot feature importances
        return feature_importances

    def save_model_metadata(self, metadata: Dict):
        """
        Save model metadata to a JSON file.

        Args:
            metadata: Dictionary containing metadata for each trained segment model.
        """
        metadata_path = self.model_dir / "model_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Model metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")


# Example usage (if run directly)
if __name__ == "__main__":
    # This part would require sample data and config to run
    logger.info("ModelTrainer script executed directly (example usage).")
    # Example: Load config, data, preprocessor, then instantiate and run trainer
    # config = load_config(...)
    # segment_data_train = load_training_data(...)
    # preprocessor = load_preprocessor(...)
    # trainer = ModelTrainer(config)
    # trained_models = trainer.train_segment_models(segment_data_train, preprocessor)
    # X_test, y_test, test_masks = load_test_data(...)
    # predictions, combined_metrics, segment_metrics = trainer.evaluate_models(trained_models, X_test, y_test, test_masks)
    # feature_importances = trainer.analyze_feature_importance(trained_models, segment_data_train)
    # trainer.save_models(trained_models) # Assuming a save_models method exists
