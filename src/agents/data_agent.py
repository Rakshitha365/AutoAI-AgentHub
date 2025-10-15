import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import logging
import os
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
from src.core.dataclasses import ProcessedPayload


class DataAgent:
    """Agent responsible for data preprocessing and preparation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the Data Agent.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.max_file_size_mb = config.get("max_file_size_mb", 10)
        self.test_size = config.get("default_test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        self.max_categorical_cardinality = config.get("max_categorical_cardinality", 50)
        self.artifact_dir = config.get("artifact_dir", "artifacts")
        self.models_dir = os.path.join(self.artifact_dir, "models")
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load(self, path_or_buffer: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load dataset from CSV file or DataFrame.
        
        Args:
            path_or_buffer: Path to CSV file or DataFrame
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If file format is invalid or file is too large
            FileNotFoundError: If file doesn't exist
        """
        self.logger.info(f"DataAgent: Loading dataset from {path_or_buffer}")
        
        if isinstance(path_or_buffer, pd.DataFrame):
            df = path_or_buffer
        else:
            # Validate file path
            if not os.path.exists(path_or_buffer):
                self.logger.error(f"DataAgent: File not found at {path_or_buffer}")
                raise FileNotFoundError(f"Dataset not found: {path_or_buffer}")
            
            # Check file size
            file_size_mb = os.path.getsize(path_or_buffer) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                self.logger.error(f"DataAgent: File size ({file_size_mb:.2f} MB) exceeds limit ({self.max_file_size_mb} MB)")
                raise ValueError(f"File size exceeds maximum allowed size of {self.max_file_size_mb} MB")
            
            # Load CSV file
            try:
                df = pd.read_csv(path_or_buffer)
            except Exception as e:
                self.logger.error(f"DataAgent: Error loading CSV file: {str(e)}")
                raise ValueError(f"Invalid CSV file format: {str(e)}")
        
        # Validate DataFrame
        if df.empty:
            self.logger.error("DataAgent: Dataset is empty")
            raise ValueError("Dataset is empty")
        
        if len(df.columns) < 2:
            self.logger.error("DataAgent: Dataset must have at least 2 columns")
            raise ValueError("Dataset must have at least 2 columns")
        
        self.logger.info(f"DataAgent: Dataset loaded successfully - {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset and generate metadata.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing dataset metadata
        """
        self.logger.info("DataAgent: Analyzing dataset for metadata")
        
        # Basic information
        meta = {
            "n_samples": df.shape[0],
            "n_features": df.shape[1],
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict(),
            "categorical_columns": [],
            "numerical_columns": [],
            "target_candidates": []
        }
        
        # Detect column types
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < len(df) * 0.1:
                meta["categorical_columns"].append(col)
            else:
                meta["numerical_columns"].append(col)
        
        # Identify potential target columns
        for col in df.columns:
            if col.lower() in ['target', 'label', 'class', 'y', 'outcome', 'result']:
                meta["target_candidates"].append(col)
            elif df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                meta["target_candidates"].append(col)
        
        self.logger.info(f"DataAgent: Analysis complete - {len(meta['categorical_columns'])} categorical, {len(meta['numerical_columns'])} numerical columns")
        return meta
    
    def preprocess(self, df: pd.DataFrame, target_col: str) -> ProcessedPayload:
        """Preprocess dataset including imputation, encoding, and scaling.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            ProcessedPayload containing train/test splits and metadata
        """
        self.logger.info(f"DataAgent: Preprocessing data with target column '{target_col}'")
        
        if target_col not in df.columns:
            self.logger.error(f"DataAgent: Target column '{target_col}' not found in DataFrame")
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Analyze features
        meta = self.analyze(X)
        meta["target_column"] = target_col
        
        # Determine task type
        if y.nunique() < len(y) * 0.1:  # Less than 10% unique values
            meta["task_type"] = "classification"
        else:
            meta["task_type"] = "regression"
        
        self.logger.info(f"DataAgent: Detected task type: {meta['task_type']}")
        
        # Prepare preprocessing pipelines
        numerical_cols = meta["numerical_columns"]
        categorical_cols = meta["categorical_columns"]
        
        # Numerical preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform data
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                # Get feature names from fitted OneHotEncoder
                try:
                    onehot_transformer = transformer.named_steps['onehot']
                    if hasattr(onehot_transformer, 'get_feature_names_out'):
                        feature_names.extend(onehot_transformer.get_feature_names_out(cols))
                    else:
                        # Fallback for older sklearn versions
                        feature_names.extend([f"{c}_{val}" for c in cols for val in X[c].dropna().unique()])
                except Exception as e:
                    self.logger.warning(f"Could not extract categorical feature names: {e}")
                    # Fallback: use original column names
                    feature_names.extend(cols)
            elif name == 'remainder' and cols != 'drop':
                feature_names.extend(X.columns[preprocessor.output_indices_['remainder']].tolist())
        
        # Create DataFrame with proper column names
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Split data
        stratify = y if meta["task_type"] == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed_df, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Save preprocessor
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        preprocessor_path = os.path.join(self.models_dir, f"preprocessor_{timestamp}.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        
        # Update metadata
        meta["feature_names"] = feature_names
        meta["preprocessor_path"] = preprocessor_path
        
        self.logger.info("DataAgent: Data preprocessing completed successfully")
        
        return ProcessedPayload(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor_path=preprocessor_path,
            meta=meta
        )
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data quality metrics
        """
        self.logger.info("DataAgent: Validating data quality")
        
        # Calculate quality metrics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_cols)) * 100 if total_rows > 0 else 0
        
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Calculate quality score (0-100)
        quality_score = 100 - min(missing_percentage, 30) - min(duplicate_percentage, 20)
        quality_score = max(quality_score, 0)
        
        quality_report = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "missing_values_count": missing_values,
            "missing_values_percentage": missing_percentage,
            "duplicate_rows_count": duplicate_rows,
            "duplicate_rows_percentage": duplicate_percentage,
            "quality_score": quality_score,
            "quality_grade": self._get_quality_grade(quality_score)
        }
        
        self.logger.info(f"DataAgent: Quality score: {quality_score:.1f}/100 ({quality_report['quality_grade']})")
        return quality_report
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score."""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"
