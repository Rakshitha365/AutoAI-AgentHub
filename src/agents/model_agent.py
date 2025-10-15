import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from src.core.dataclasses import ProcessedPayload, ModelArtifact


class ModelAgent:
    """Agent responsible for model training, evaluation, and selection."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the Model Agent.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.models_dir = os.path.join(config.get("artifact_dir", "artifacts"), "models")
        self.random_state = config.get("random_state", 42)
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize model pools
        self.classification_models = {
            "LogisticRegression": LogisticRegression(random_state=self.random_state, solver='liblinear'),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=self.random_state),
            "RandomForestClassifier": RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        }
        
        self.regression_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=self.random_state),
            "RandomForestRegressor": RandomForestRegressor(random_state=self.random_state, n_estimators=100)
        }
        
        self.logger.info("ModelAgent: Initialized with model pools")
    
    def train_and_evaluate(self, payload: ProcessedPayload) -> ModelArtifact:
        """Train and evaluate models based on processed data.
        
        Args:
            payload: ProcessedPayload containing train/test data
            
        Returns:
            ModelArtifact containing the best model and metrics
        """
        self.logger.info("ModelAgent: Starting model training and evaluation")
        
        task_type = payload.meta.get("task_type")
        if not task_type:
            self.logger.error("ModelAgent: Task type not defined in metadata")
            raise ValueError("Task type (classification/regression) must be defined in meta")
        
        # Select appropriate models
        if task_type == "classification":
            models_to_try = self.classification_models
        elif task_type == "regression":
            models_to_try = self.regression_models
        else:
            self.logger.error(f"ModelAgent: Unknown task type '{task_type}'")
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Train and evaluate all models
        results = []
        for model_name, model_instance in models_to_try.items():
            self.logger.info(f"ModelAgent: Training {model_name}...")
            
            try:
                # Train model
                trained_model = self._train_model(model_instance, payload.X_train, payload.y_train)
                
                # Evaluate model
                metrics = self._evaluate_model(trained_model, payload.X_test, payload.y_test, task_type)
                
                self.logger.info(f"ModelAgent: {model_name} - {self._format_metrics(metrics)}")
                
                results.append({
                    "model_name": model_name,
                    "model_instance": trained_model,
                    "metrics": metrics
                })
                
            except Exception as e:
                self.logger.error(f"ModelAgent: Error training {model_name}: {str(e)}")
                continue
        
        if not results:
            self.logger.error("ModelAgent: No models were successfully trained")
            raise RuntimeError("Failed to train any models")
        
        # Select best model
        best_model_info = self._select_best_model(results, task_type)
        if not best_model_info:
            self.logger.error("ModelAgent: No best model could be selected")
            raise RuntimeError("Failed to select a best model")
        
        # Save best model
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = os.path.join(self.models_dir, f"best_model_{timestamp}.joblib")
        joblib.dump(best_model_info["model_instance"], model_path)
        
        self.logger.info(f"ModelAgent: Best model ({best_model_info['model_name']}) saved to {model_path}")
        
        # Create model artifact
        model_artifact = ModelArtifact(
            model_path=model_path,
            model_type=best_model_info["model_name"],
            metrics=best_model_info["metrics"],
            timestamp=datetime.now().isoformat(),
            preprocessor_path=payload.preprocessor_path,
            task_type=task_type,
            feature_names=payload.meta.get("feature_names", []),
            target_column=payload.meta.get("target_column", "")
        )
        
        return model_artifact
    
    def _train_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a single model.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        model.fit(X_train, y_train)
        return model
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> Dict[str, float]:
        """Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            task_type: Type of task (classification/regression)
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = model.predict(X_test)
        metrics = {}
        
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, predictions)
            metrics["f1_score"] = f1_score(y_test, predictions, average='weighted', zero_division=0)
            metrics["precision"] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            metrics["recall"] = recall_score(y_test, predictions, average='weighted', zero_division=0)
        elif task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, predictions)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, predictions)
            metrics["r2_score"] = r2_score(y_test, predictions)
        
        return metrics
    
    def _select_best_model(self, results: List[Dict[str, Any]], task_type: str) -> Optional[Dict[str, Any]]:
        """Select the best model from evaluation results.
        
        Args:
            results: List of model evaluation results
            task_type: Type of task (classification/regression)
            
        Returns:
            Best model information or None
        """
        if not results:
            return None
        
        if task_type == "classification":
            # For classification, prioritize F1-score, then accuracy
            best_model = max(results, key=lambda x: (
                x["metrics"].get("f1_score", 0),
                x["metrics"].get("accuracy", 0)
            ))
        elif task_type == "regression":
            # For regression, prioritize lower RMSE, then higher R2
            best_model = min(results, key=lambda x: (
                x["metrics"].get("rmse", float('inf')),
                -x["metrics"].get("r2_score", -float('inf'))
            ))
        else:
            return None
        
        return best_model
    
    def save_model_metrics(self, metrics: Dict[str, float], metrics_path: str) -> str:
        """Save model metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            metrics_path: Path where to save the metrics
            
        Returns:
            Path where metrics were saved
        """
        try:
            import json
            import os
            # Ensure directory exists
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            self.logger.info(f"Metrics saved to {metrics_path}")
            return metrics_path
        except Exception as e:
            self.logger.error(f"Error saving metrics to {metrics_path}: {e}")
            raise
    
    def load_model(self, model_path: str) -> Any:
        """Load a trained model from disk.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        self.logger.info(f"ModelAgent: Loading model from {model_path}")
        return joblib.load(model_path)
    
    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        return model.predict(X)
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                coefs = coefs.mean(axis=0)  # Average across classes
            importance_dict = dict(zip(feature_names, coefs))
        else:
            self.logger.warning("ModelAgent: Model does not support feature importance")
        
        return importance_dict
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for logging."""
        formatted = []
        for metric, value in metrics.items():
            formatted.append(f"{metric}: {value:.4f}")
        return ", ".join(formatted)
