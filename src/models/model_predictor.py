import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Class for loading trained models and making predictions.
    Handles segment-specific models and preprocessing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ModelPredictor with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.model_dir = Path(config['paths']['model_dir'])
        self.segment_models = {}
        self.preprocessor = None
        self.premium_thresholds = None
        self.metadata = None
        
    def load_models(self, version: Optional[str] = None) -> bool:
        """
        Load segment models, preprocessor, and metadata.
        
        Args:
            version: Optional version identifier. If None, loads the latest version.
            
        Returns:
            Boolean indicating if loading was successful
        """
        try:
            # Determine version to load
            if version is None:
                # Find latest version
                versions = [d for d in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, d))]
                if not versions:
                    logger.error("No model versions found.")
                    return False
                version = sorted(versions)[-1]  # Get latest version
            
            version_dir = self.model_dir / version
            
            # Load metadata
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            
            # Load premium thresholds
            thresholds_path = version_dir / "premium_thresholds.json"
            with open(thresholds_path, "r") as f:
                self.premium_thresholds = json.load(f)
            
            # Load preprocessor
            preprocessor_path = version_dir / "preprocessor.joblib"
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Load segment models
            self.segment_models = {}
            for segment in ["very_low", "low", "medium", "high", "very_high"]:
                model_path = version_dir / f"{segment}_model.joblib"
                if os.path.exists(model_path):
                    self.segment_models[segment] = joblib.load(model_path)
                else:
                    logger.warning(f"Model for segment '{segment}' not found.")
            
            if not self.segment_models:
                logger.error("No segment models were loaded.")
                return False
                
            logger.info(f"Successfully loaded {len(self.segment_models)} segment models from version {version}.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _validate_input(self, X: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate input data format and required columns.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if DataFrame
        if not isinstance(X, pd.DataFrame):
            return False, "Input must be a pandas DataFrame."
            
        # Check required columns (using a sample from the dataset)
        required_columns = self.config.get('data', {}).get('required_columns', [])
        if required_columns:
            missing_columns = [col for col in required_columns if col not in X.columns]
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check if empty
        if X.empty:
            return False, "Input DataFrame is empty."
            
        return True, None
    
    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to the input data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Clean column names
        X.columns = [col.replace(' ', '_') for col in X.columns]
        
        # Process date features
        try:
            if 'Policy_Start_Date' in X.columns:
                X['Policy_Start_Date'] = pd.to_datetime(X['Policy_Start_Date'], errors='coerce')
                
                # Extract date features
                X['Policy_Start_Year'] = X['Policy_Start_Date'].dt.year
                X['Policy_Start_Month'] = X['Policy_Start_Date'].dt.month
                X['Policy_Start_Day'] = X['Policy_Start_Date'].dt.day
                X['Policy_Start_DayOfWeek'] = X['Policy_Start_Date'].dt.dayofweek
                X['Policy_Start_Quarter'] = X['Policy_Start_Date'].dt.quarter
                X['Policy_Start_IsWeekend'] = X['Policy_Start_Date'].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Calculate days since reference date
                reference_date = pd.Timestamp('2020-01-01')
                X['Days_Since_Reference'] = (X['Policy_Start_Date'] - reference_date).dt.days
                
                # Drop original date column
                X = X.drop(columns=['Policy_Start_Date'])
        except Exception as e:
            logger.warning(f"Error processing date features: {e}")
        
        # Perform feature engineering
        # Temporarily impute missing values for feature engineering
        num_cols_for_eng = X.select_dtypes(include=np.number).columns
        
        if len(num_cols_for_eng) > 0:
            # Simple median imputation for feature engineering only
            for col in num_cols_for_eng:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
        
        # Create features
        if 'Previous_Claims' in X.columns and 'Insurance_Duration' in X.columns:
            X['Claims_per_Year'] = X['Previous_Claims'] / X['Insurance_Duration'].clip(1)
            X['Has_Claims'] = (X['Previous_Claims'] > 0).astype(int)
            
        if 'Vehicle_Age' in X.columns: 
            X['Vehicle_Age_Risk'] = np.exp(X['Vehicle_Age'] / 10)
            
        if 'Annual_Income' in X.columns:
            X['Log_Income'] = np.log1p(X['Annual_Income'])
            if 'Number_of_Dependents' in X.columns: 
                X['Income_per_Dependent'] = X['Annual_Income'] / X['Number_of_Dependents'].replace(0, 1).fillna(1)
                
        if 'Credit_Score' in X.columns: 
            X['Credit_Factor'] = np.exp((X['Credit_Score'] - 300) / 100)
        
        # Create interaction features
        key_interactions = [
            ('Age', 'Health_Score'), 
            ('Credit_Score', 'Annual_Income'), 
            ('Previous_Claims', 'Vehicle_Age'), 
            ('Age', 'Vehicle_Age'), 
            ('Credit_Score', 'Insurance_Duration'), 
            ('Log_Income', 'Credit_Factor')
        ]
        
        for col1, col2 in key_interactions:
            if col1 in X.columns and col2 in X.columns:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        return X
    
    def _determine_segment(self, predictions: np.ndarray) -> List[str]:
        """
        Determine which segment each prediction belongs to.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            List of segment labels for each prediction
        """
        if self.premium_thresholds is None:
            raise ValueError("Premium thresholds not loaded.")
            
        segments = []
        for pred in predictions:
            if pred <= self.premium_thresholds['p25']:
                segments.append('very_low')
            elif pred <= self.premium_thresholds['p50']:
                segments.append('low')
            elif pred <= self.premium_thresholds['p75']:
                segments.append('medium')
            elif pred <= self.premium_thresholds['p95']:
                segments.append('high')
            else:
                segments.append('very_high')
                
        return segments
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Make predictions using the appropriate segment model for each data point.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.segment_models:
            raise ValueError("Models not loaded. Call load_models() first.")
            
        # Validate input
        is_valid, error_message = self._validate_input(X)
        if not is_valid:
            raise ValueError(f"Invalid input: {error_message}")
            
        # Preprocess data
        X_processed = self._preprocess_data(X)
        
        # Initialize prediction array
        predictions = np.zeros(len(X_processed))
        segment_assignments = []
        
        # First, make a preliminary prediction using the medium segment model to decide segments
        medium_model = self.segment_models.get('medium')
        if medium_model is None:
            # Fall back to another segment model
            for segment, model in self.segment_models.items():
                if model is not None:
                    medium_model = model
                    logger.warning(f"Medium segment model not found. Using {segment} model for preliminary prediction.")
                    break
            else:
                raise ValueError("No valid model found for segment determination.")
        
        try:
            # Use medium model for initial prediction
            initial_predictions = medium_model.predict(X_processed)
            initial_predictions = np.maximum(0, initial_predictions)  # Ensure non-negative
            
            # Determine segments
            segments = self._determine_segment(initial_predictions)
            
            # Make predictions using the appropriate segment model for each data point
            for i, segment in enumerate(segments):
                model = self.segment_models.get(segment)
                if model is not None:
                    # Use a single row for prediction
                    row_prediction = model.predict(X_processed.iloc[[i]])
                    predictions[i] = max(0, row_prediction[0])  # Ensure non-negative
                else:
                    # Fall back to medium model or any available model
                    fallback_model = medium_model  # Already checked to be not None
                    row_prediction = fallback_model.predict(X_processed.iloc[[i]])
                    predictions[i] = max(0, row_prediction[0])
                    logger.warning(f"Model for segment '{segment}' not available. Using fallback model.")
                
                segment_assignments.append(segment)
            
            result = {
                'predictions': predictions.tolist(),
                'segments': segment_assignments,
                'success': True,
                'message': f"Successfully predicted {len(predictions)} samples."
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {
                'predictions': None,
                'segments': None,
                'success': False,
                'message': f"Error: {str(e)}"
            }
    
    def predict_batch(self, X: pd.DataFrame) -> Dict:
        """
        Make predictions for batch data using segment models.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.segment_models:
            raise ValueError("Models not loaded. Call load_models() first.")
            
        # Validate input
        is_valid, error_message = self._validate_input(X)
        if not is_valid:
            raise ValueError(f"Invalid input: {error_message}")
            
        # Preprocess data
        X_processed = self._preprocess_data(X)
        
        # First, make preliminary predictions using the medium segment model
        medium_model = self.segment_models.get('medium')
        if medium_model is None:
            # Fall back to another segment model
            for segment, model in self.segment_models.items():
                if model is not None:
                    medium_model = model
                    logger.warning(f"Medium segment model not found. Using {segment} model for preliminary prediction.")
                    break
            else:
                raise ValueError("No valid model found for segment determination.")
        
        try:
            # Use medium model for initial prediction
            initial_predictions = medium_model.predict(X_processed)
            initial_predictions = np.maximum(0, initial_predictions)  # Ensure non-negative
            
            # Determine segments
            segments = self._determine_segment(initial_predictions)
            unique_segments = set(segments)
            
            # Group data by segment and predict
            predictions = np.zeros(len(X_processed))
            
            for segment in unique_segments:
                # Create mask for this segment
                segment_mask = [i for i, s in enumerate(segments) if s == segment]
                if not segment_mask:
                    continue
                    
                # Get model for this segment
                model = self.segment_models.get(segment)
                if model is None:
                    # Fall back to medium model
                    model = medium_model
                    logger.warning(f"Model for segment '{segment}' not available. Using fallback model.")
                
                # Make predictions for this segment
                X_segment = X_processed.iloc[segment_mask]
                segment_predictions = model.predict(X_segment)
                segment_predictions = np.maximum(0, segment_predictions)  # Ensure non-negative
                
                # Assign predictions to the correct indices
                for idx, pred_idx in enumerate(segment_mask):
                    predictions[pred_idx] = segment_predictions[idx]
            
            result = {
                'predictions': predictions.tolist(),
                'segments': segments,
                'success': True,
                'message': f"Successfully predicted {len(predictions)} samples across {len(unique_segments)} segments."
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return {
                'predictions': None,
                'segments': None,
                'success': False,
                'message': f"Error: {str(e)}"
            }

