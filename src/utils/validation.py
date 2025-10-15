import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 2) -> Tuple[bool, str]:
    """Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        min_cols: Minimum number of columns required
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"
    
    if len(df.columns) < min_cols:
        return False, f"DataFrame has {len(df.columns)} columns, minimum {min_cols} required"
    
    return True, "DataFrame is valid"


def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate data quality score and metrics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing quality metrics
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Calculate missing values
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values / (total_rows * total_cols)) * 100 if total_rows > 0 else 0
    
    # Calculate duplicates
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
    
    # Calculate quality score
    quality_score = 100 - min(missing_percentage, 30) - min(duplicate_percentage, 20)
    quality_score = max(quality_score, 0)
    
    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_values_count": missing_values,
        "missing_values_percentage": missing_percentage,
        "duplicate_rows_count": duplicate_rows,
        "duplicate_rows_percentage": duplicate_percentage,
        "quality_score": quality_score,
        "quality_grade": get_quality_grade(quality_score)
    }


def get_quality_grade(score: float) -> str:
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


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect column types in DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with column type lists
    """
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < len(df) * 0.1:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    return {
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols
    }


def create_directory_structure(base_path: str, directories: List[str]) -> None:
    """Create directory structure.
    
    Args:
        base_path: Base directory path
        directories: List of directory names to create
    """
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metrics for display.
    
    Args:
        metrics: Metrics dictionary
        precision: Decimal precision
        
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    for metric, value in metrics.items():
        if isinstance(value, float):
            formatted[metric] = f"{value:.{precision}f}"
        else:
            formatted[metric] = str(value)
    return formatted


def generate_timestamp() -> str:
    """Generate timestamp string."""
    return pd.Timestamp.now().strftime('%Y%m%d%H%M%S')


def validate_file_path(file_path: str, max_size_mb: int = 10) -> Tuple[bool, str]:
    """Validate file path and size.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)"
    
    return True, "File is valid"


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("AutoAI")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def create_sample_data(n_samples: int = 100, n_features: int = 5, 
                      task_type: str = 'classification') -> pd.DataFrame:
    """Create sample data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: Type of task (classification/regression)
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Generate features
    data = {}
    for i in range(n_features):
        if i % 3 == 0:  # Numerical features
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        elif i % 3 == 1:  # Categorical features
            data[f'feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
        else:  # Binary features
            data[f'feature_{i}'] = np.random.choice([0, 1], n_samples)
    
    # Generate target
    if task_type == 'classification':
        data['target'] = np.random.choice([0, 1], n_samples)
    else:
        data['target'] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)
