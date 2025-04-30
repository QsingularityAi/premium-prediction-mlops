import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import logging
import os
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class responsible for data loading, preprocessing, and feature engineering
    for the premium prediction model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataProcessor with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.target_column = config['data']['target_column']
        self.random_state = config['training']['random_state']
        self.test_size = config['data']['test_size']  # Already correctly accessing from data section
        self.data_path = config['data']['data_path']
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the dataset from the specified path.
        
        Args:
            data_path: Path to the data file (overrides config path if provided)
            
        Returns:
            DataFrame containing the loaded data
        """
        path = data_path or self.data_path
        try:
            df = pd.read_csv(path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"Error: '{path}' not found.")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data preprocessing steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Drop ID column if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            logger.info("Dropped 'id' column.")
            
        # Clean column names
        df.columns = [col.replace(' ', '_') for col in df.columns]
        logger.info("Cleaned column names.")
        
        # Process date features
        try:
            if 'Policy_Start_Date' in df.columns:
                df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'], errors='coerce')
                orig_count = len(df)
                df.dropna(subset=['Policy_Start_Date'], inplace=True)
                if orig_count > len(df): 
                    logger.info(f"Dropped {orig_count - len(df)} rows with invalid dates.")
                
                # Extract date features
                df['Policy_Start_Year'] = df['Policy_Start_Date'].dt.year
                df['Policy_Start_Month'] = df['Policy_Start_Date'].dt.month
                df['Policy_Start_Day'] = df['Policy_Start_Date'].dt.day
                df['Policy_Start_DayOfWeek'] = df['Policy_Start_Date'].dt.dayofweek
                df['Policy_Start_Quarter'] = df['Policy_Start_Date'].dt.quarter
                df['Policy_Start_IsWeekend'] = df['Policy_Start_Date'].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Calculate days since reference date
                reference_date = pd.Timestamp('2020-01-01')
                df['Days_Since_Reference'] = (df['Policy_Start_Date'] - reference_date).dt.days
                
                # Drop original date column
                df = df.drop(columns=['Policy_Start_Date'])
                logger.info("Created enhanced date features.")
            else:
                logger.info("Policy_Start_Date column not found, skipping date features.")
        except Exception as e:
            logger.error(f"Error processing date features: {e}")
        
        # Handle target variable
        if self.target_column not in df.columns:
            logger.error(f"Error: Target column '{self.target_column}' not found.")
            raise ValueError(f"Target column '{self.target_column}' not found.")
            
        df.dropna(subset=[self.target_column], inplace=True)
        df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
        df.dropna(subset=[self.target_column], inplace=True)
        
        # Cap target variable outliers at 99th percentile
        p99 = df[self.target_column].quantile(0.99)
        outliers_count = sum(df[self.target_column] > p99)
        if outliers_count > 0:
            logger.info(f"Capping {outliers_count} outliers at 99th percentile ({p99:.2f}).")
            df[self.target_column] = np.where(df[self.target_column] > p99, p99, df[self.target_column])
            
        return df
    
    def perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Performing feature engineering...")
        
        # Temporarily impute missing values for feature engineering
        temp_imputer_median = SimpleImputer(strategy='median')
        num_cols_for_eng = df.select_dtypes(include=np.number).drop(columns=[self.target_column]).columns
        
        if len(num_cols_for_eng) > 0:
            df[num_cols_for_eng] = temp_imputer_median.fit_transform(df[num_cols_for_eng])
        else:
            logger.warning("Warning: No numeric columns found for temporary imputation during FE.")
        
        # Create features
        if 'Previous_Claims' in df.columns and 'Insurance_Duration' in df.columns:
            df['Claims_per_Year'] = df['Previous_Claims'] / df['Insurance_Duration'].clip(1)
            df['Has_Claims'] = (df['Previous_Claims'] > 0).astype(int)
            
        if 'Vehicle_Age' in df.columns: 
            df['Vehicle_Age_Risk'] = np.exp(df['Vehicle_Age'] / 10)
            
        if 'Annual_Income' in df.columns:
            df['Log_Income'] = np.log1p(df['Annual_Income'])
            if 'Number_of_Dependents' in df.columns: 
                df['Income_per_Dependent'] = df['Annual_Income'] / df['Number_of_Dependents'].replace(0, 1).fillna(1)
                
        if 'Credit_Score' in df.columns: 
            df['Credit_Factor'] = np.exp((df['Credit_Score'] - 300) / 100)
        
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
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into features and target, then into training and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Identify feature types
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        
        # Ensure no overlap and select final features
        numerical_features = [f for f in numerical_features if f in X.columns]
        categorical_features = [f for f in categorical_features if f in X.columns]
        final_features = numerical_features + categorical_features
        X = X[final_features]
        
        logger.info(f"Identified {len(categorical_features)} categorical features.")
        logger.info(f"Identified {len(numerical_features)} numerical features.")
        logger.info(f"Total features selected: {X.shape[1]}")
        
        # Split data
        logger.info("Splitting data...")
        try:
            target_bins = pd.qcut(y, 4, labels=False, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state, 
                stratify=target_bins
            )
        except ValueError as e:
            logger.warning(f"Warning: Stratified split failed ('{e}'). Using non-stratified.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self, numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for numerical and categorical features.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            
        Returns:
            ColumnTransformer preprocessing pipeline
        """
        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', ce.CatBoostEncoder(handle_unknown='value', handle_missing='value'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        
        preprocessor.set_output(transform="pandas")
        return preprocessor
    
    def create_segments(self, y_train: pd.Series) -> Tuple[Dict, Dict]:
        """
        Define premium segments based on quantiles of the target variable.
        
        Args:
            y_train: Training data target variable
            
        Returns:
            Tuple containing (premium_thresholds, segment_data_train)
        """
        logger.info("Defining premium segments...")
        
        # Define thresholds
        p25, p50, p75, p95 = y_train.quantile([0.25, 0.50, 0.75, 0.95])
        premium_thresholds = {'p25': p25, 'p50': p50, 'p75': p75, 'p95': p95}
        logger.info(f"Thresholds: p25={p25:.2f}, p50={p50:.2f}, p75={p75:.2f}, p95={p95:.2f}")
        
        return premium_thresholds
    
    def split_into_segments(self, X_train: pd.DataFrame, y_train: pd.Series, premium_thresholds: Dict) -> Dict:
        """
        Split training data into segments based on premium thresholds.
        
        Args:
            X_train: Training features
            y_train: Training target
            premium_thresholds: Dictionary of premium thresholds
            
        Returns:
            Dictionary mapping segment names to (X, y) tuples
        """
        # Create masks for each segment
        very_low_mask_train = y_train <= premium_thresholds['p25']
        low_mask_train = (y_train > premium_thresholds['p25']) & (y_train <= premium_thresholds['p50'])
        medium_mask_train = (y_train > premium_thresholds['p50']) & (y_train <= premium_thresholds['p75'])
        high_mask_train = (y_train > premium_thresholds['p75']) & (y_train <= premium_thresholds['p95'])
        very_high_mask_train = y_train > premium_thresholds['p95']
        
        # Create dictionary of segment data
        segment_data_train = {
            'very_low': (X_train[very_low_mask_train], y_train[very_low_mask_train]),
            'low': (X_train[low_mask_train], y_train[low_mask_train]),
            'medium': (X_train[medium_mask_train], y_train[medium_mask_train]),
            'high': (X_train[high_mask_train], y_train[high_mask_train]),
            'very_high': (X_train[very_high_mask_train], y_train[very_high_mask_train])
        }
        
        # Log segment sizes
        for segment, (X_seg, _) in segment_data_train.items():
            logger.info(f"{segment.capitalize()} train segment: {len(X_seg)} samples")
            
        return segment_data_train
    
    def get_segment_masks_test(self, y_test: pd.Series, premium_thresholds: Dict) -> Dict:
        """
        Create segment masks for test data.
        
        Args:
            y_test: Test target data
            premium_thresholds: Dictionary of premium thresholds
            
        Returns:
            Dictionary of test data masks for each segment
        """
        # Create masks for test data
        very_low_mask_test = y_test <= premium_thresholds['p25']
        low_mask_test = (y_test > premium_thresholds['p25']) & (y_test <= premium_thresholds['p50'])
        medium_mask_test = (y_test > premium_thresholds['p50']) & (y_test <= premium_thresholds['p75'])
        high_mask_test = (y_test > premium_thresholds['p75'])
        very_high_mask_test = y_test > premium_thresholds['p95']
        
        # Create dictionary of segment masks
        segment_masks_test = {
            'very_low': very_low_mask_test,
            'low': low_mask_test,
            'medium': medium_mask_test,
            'high': high_mask_test,
            'very_high': very_high_mask_test
        }
        
        return segment_masks_test
