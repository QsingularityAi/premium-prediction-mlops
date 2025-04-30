import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib
import tempfile
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import shutil

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_trainer import ModelTrainer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class TestModelTrainer(unittest.TestCase):
    """Tests for the ModelTrainer class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a temp directory for models
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock configuration
        self.config = {
            'training': {
                'random_state': 42,
                'n_iter_search': 2,  # Small value for testing
                'cv_folds_tuning': 2,
                'cv_folds_eval': 2,
                'feature_selection_threshold': 'median'
            },
            'paths': {
                'model_dir': self.test_dir
            },
            'segments': {
                'very_low': {'model_type': 'lgbm', 'use_log_transform': False},
                'low': {'model_type': 'lgbm', 'use_log_transform': False},
                'medium': {'model_type': 'lgbm', 'use_log_transform': False},
                'high': {'model_type': 'lgbm', 'use_log_transform': False},
                'very_high': {'model_type': 'lgbm', 'use_log_transform': False}
            }
        }
        
        # Create test data
        # X features for each segment
        self.X_train = pd.DataFrame({
            'Feature1': np.random.rand(10),
            'Feature2': np.random.rand(10),
            'Feature3': np.random.rand(10)
        })
        
        # Y target for each segment
        self.y_train = pd.Series(np.random.rand(10) * 1000)
        
        # X test data
        self.X_test = pd.DataFrame({
            'Feature1': np.random.rand(5),
            'Feature2': np.random.rand(5),
            'Feature3': np.random.rand(5)
        })
        
        # Y test data
        self.y_test = pd.Series(np.random.rand(5) * 1000)
        
        # Create segment data
        self.segment_data_train = {
            'very_low': (self.X_train.iloc[:2], self.y_train.iloc[:2]),
            'low': (self.X_train.iloc[2:4], self.y_train.iloc[2:4]),
            'medium': (self.X_train.iloc[4:6], self.y_train.iloc[4:6]),
            'high': (self.X_train.iloc[6:8], self.y_train.iloc[6:8]),
            'very_high': (self.X_train.iloc[8:], self.y_train.iloc[8:])
        }
        
        # Create test masks
        self.test_masks = {
            'very_low': np.array([True, True, False, False, False]),
            'low': np.array([False, False, True, True, False]),
            'medium': np.array([False, False, False, False, True]),
            'high': np.array([False, False, False, False, False]),
            'very_high': np.array([False, False, False, False, False])
        }
        
        # Create a simple preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), ['Feature1', 'Feature2', 'Feature3'])
            ]
        )
        
        # Initialize model trainer
        self.model_trainer = ModelTrainer(self.config)
        
    def tearDown(self):
        """Clean up after each test method"""
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test ModelTrainer initialization"""
        self.assertEqual(self.model_trainer.random_state, 42)
        self.assertEqual(self.model_trainer.n_iter_search, 2)
        self.assertEqual(self.model_trainer.cv_folds_tuning, 2)
        self.assertEqual(self.model_trainer.cv_folds_eval, 2)
        self.assertEqual(self.model_trainer.feature_selection_threshold, 'median')
        self.assertEqual(str(self.model_trainer.model_dir), str(Path(self.test_dir)))
        
    def test_define_model_configs(self):
        """Test the model configuration definitions"""
        model_configs = self.model_trainer.define_model_configs(self.segment_data_train)
        
        # Check if configurations are created for all segments
        for segment in self.segment_data_train.keys():
            self.assertIn(segment, model_configs)
            
        # Check structure of configs
        for segment, config in model_configs.items():
            self.assertIn('model', config)
            self.assertIn('selector', config)
            self.assertIn('params', config)
            self.assertIn('use_log_transform', config)
            
            # Check if params contain expected hyperparameters
            self.assertIn('model__n_estimators', config['params'])
            self.assertIn('model__learning_rate', config['params'])
        
    @patch('mlflow.sklearn.log_model')
    def test_train_segment_models(self, mock_log_model):
        """Test the segment model training"""
        # Let's avoid mocking the entire training process and
        # instead check that the models are returned correctly
        
        # Create a simpler test by adding fixed hyperparameters to config
        # to avoid RandomizedSearchCV altogether
        self.config['segments'] = {
            'very_low': {'model_type': 'lgbm', 'use_log_transform': False, 'hyperparams': {
                'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31
            }},
            'low': {'model_type': 'lgbm', 'use_log_transform': False, 'hyperparams': {
                'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31
            }},
            'medium': {'model_type': 'lgbm', 'use_log_transform': False, 'hyperparams': {
                'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31
            }},
            'high': {'model_type': 'lgbm', 'use_log_transform': False, 'hyperparams': {
                'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31
            }},
            'very_high': {'model_type': 'lgbm', 'use_log_transform': False, 'hyperparams': {
                'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 31
            }}
        }
        
        # Create a mock for the segments data with more samples
        large_segment_data = {}
        for segment in self.segment_data_train.keys():
            # We'll keep the same X_train but multiply it by 5 rows to get sufficient data
            X = pd.concat([self.X_train] * 5, ignore_index=True)
            y = pd.concat([pd.Series(self.y_train)] * 5, ignore_index=True)
            large_segment_data[segment] = (X, y)
        
        # Call the method under test
        segment_model_configs = self.model_trainer.define_model_configs(self.segment_data_train)
        trained_models = self.model_trainer.train_segment_models(
            large_segment_data,  # Use our larger dataset 
            self.preprocessor,
            segment_model_configs
        )
        
        # Verify models were trained for each segment
        for segment in large_segment_data.keys():
            self.assertIn(segment, trained_models)
            # Check if a model was returned (not None)
            self.assertIsNotNone(trained_models[segment])
            
    def test_evaluate_models(self):
        """Test model evaluation"""
        # Create mocked trained models
        trained_models = {}
        for segment in self.segment_data_train.keys():
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([500.0] * 5)  # Constant prediction for testing
            trained_models[segment] = mock_model
        
        # Call the method under test
        y_pred, combined_metrics, segment_metrics = self.model_trainer.evaluate_models(
            trained_models, 
            self.X_test,
            self.y_test,
            self.test_masks
        )
        
        # Verify predictions were made
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Verify combined metrics
        self.assertIn('rmse', combined_metrics)
        self.assertIn('mae', combined_metrics)
        self.assertIn('r2', combined_metrics)
        
        # Verify segment metrics
        for segment in self.segment_data_train.keys():
            if np.any(self.test_masks[segment]):  # If segment has test samples
                self.assertIn(segment, segment_metrics)
                self.assertIn('rmse', segment_metrics[segment])
                self.assertIn('mae', segment_metrics[segment])
                self.assertIn('r2', segment_metrics[segment])
                self.assertIn('count', segment_metrics[segment])
    
    def test_analyze_feature_importance(self):
        """Test feature importance analysis"""
        # Create mocked trained models
        trained_models = {}
        for segment in self.segment_data_train.keys():
            mock_model = MagicMock()
            trained_models[segment] = mock_model
        
        # Call the method under test
        result = self.model_trainer.analyze_feature_importance(
            trained_models,
            self.segment_data_train,
            self.test_dir
        )
        
        # The current implementation returns an empty dict
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()

