from dataclasses import dataclass
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class ProcessedPayload:
    """Data structure containing processed dataset splits and metadata."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor_path: str
    meta: Dict[str, Any]


@dataclass
class ModelArtifact:
    """Data structure containing trained model and associated metadata."""
    model_path: str
    model_type: str
    metrics: Dict[str, float]
    timestamp: str
    preprocessor_path: str
    task_type: str
    feature_names: List[str]
    target_column: str


@dataclass
class DeploymentInfo:
    """Data structure containing deployment information and configuration."""
    app_path: str
    app_url: str
    model_artifact: ModelArtifact
    deployment_timestamp: str
    status: str
    port: Optional[int] = None


@dataclass
class AgentResult:
    """Generic result structure for agent operations."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None


@dataclass
class Configuration:
    """Configuration data structure for the entire system."""
    max_file_size_mb: int = 10
    default_test_size: float = 0.2
    random_state: int = 42
    max_categorical_cardinality: int = 50
    streamlit_port: int = 8501
    log_level: str = "INFO"
    artifact_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    logs_dir: str = "artifacts/logs"
    reports_dir: str = "artifacts/reports"
    model_trials: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.model_trials is None:
            self.model_trials = {
                "classification": [
                    "LogisticRegression",
                    "DecisionTreeClassifier", 
                    "RandomForestClassifier"
                ],
                "regression": [
                    "LinearRegression",
                    "Ridge",
                    "RandomForestRegressor"
                ]
            }
