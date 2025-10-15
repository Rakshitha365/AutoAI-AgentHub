import pytest
import pandas as pd
import numpy as np
import tempfile
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.model_agent import ModelAgent
from src.core.dataclasses import ProcessedPayload


class TestModelAgent:
    """Test cases for Model Agent."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'model_trials': {
                'classification': ['LogisticRegression', 'DecisionTreeClassifier'],
                'regression': ['LinearRegression', 'Ridge']
            },
            'random_state': 42
        }
    
    @pytest.fixture
    def sample_logger(self):
        """Sample logger for testing."""
        return logging.getLogger('test')
    
    @pytest.fixture
    def model_agent(self, sample_config, sample_logger):
        """Model Agent instance for testing."""
        return ModelAgent(sample_config, sample_logger)
    
    @pytest.fixture
    def sample_classification_data(self):
        """Sample classification data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20)
        })
        y_train = np.random.choice([0, 1], 100)
        y_test = np.random.choice([0, 1], 20)
        
        return ProcessedPayload(
            X_train=X_train,
            X_test=X_test,
            y_train=pd.Series(y_train),
            y_test=pd.Series(y_test),
            preprocessor_path="test_preprocessor.joblib",
            meta={'task_type': 'classification', 'target_column': 'target'}
        )
    
    @pytest.fixture
    def sample_regression_data(self):
        """Sample regression data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20)
        })
        y_train = np.random.normal(0, 1, 100)
        y_test = np.random.normal(0, 1, 20)
        
        return ProcessedPayload(
            X_train=X_train,
            X_test=X_test,
            y_train=pd.Series(y_train),
            y_test=pd.Series(y_test),
            preprocessor_path="test_preprocessor.joblib",
            meta={'task_type': 'regression', 'target_column': 'target'}
        )
    
    def test_model_agent_initialization(self, model_agent):
        """Test Model Agent initialization."""
        assert model_agent is not None
        assert 'LogisticRegression' in model_agent.classification_models
        assert 'LinearRegression' in model_agent.regression_models
        assert model_agent.random_state == 42
    
    def test_classification_model_training(self, model_agent, sample_classification_data):
        """Test classification model training."""
        model_artifact = model_agent.train_and_evaluate(sample_classification_data)
        
        assert model_artifact is not None
        assert model_artifact.task_type == 'classification'
        assert model_artifact.model_type in ['LogisticRegression', 'DecisionTreeClassifier']
        assert 'accuracy' in model_artifact.metrics
        assert 'f1_score' in model_artifact.metrics
        assert model_artifact.metrics['accuracy'] >= 0
        assert model_artifact.metrics['accuracy'] <= 1
    
    def test_regression_model_training(self, model_agent, sample_regression_data):
        """Test regression model training."""
        model_artifact = model_agent.train_and_evaluate(sample_regression_data)
        
        assert model_artifact is not None
        assert model_artifact.task_type == 'regression'
        assert model_artifact.model_type in ['LinearRegression', 'Ridge']
        assert 'mse' in model_artifact.metrics
        assert 'r2_score' in model_artifact.metrics
        assert model_artifact.metrics['mse'] >= 0
    
    def test_model_evaluation_classification(self, model_agent):
        """Test model evaluation for classification."""
        from sklearn.linear_model import LogisticRegression
        
        # Create sample data
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        y_test = pd.Series([0, 1, 0])
        
        # Train a simple model
        model = LogisticRegression(random_state=42)
        model.fit(X_test, y_test)
        
        # Evaluate
        metrics = model_agent._evaluate_model(model, X_test, y_test, 'classification')
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, float))
    
    def test_model_evaluation_regression(self, model_agent):
        """Test model evaluation for regression."""
        from sklearn.linear_model import LinearRegression
        
        # Create sample data
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        y_test = pd.Series([1.1, 2.2, 3.3])
        
        # Train a simple model
        model = LinearRegression()
        model.fit(X_test, y_test)
        
        # Evaluate
        metrics = model_agent._evaluate_model(model, X_test, y_test, 'regression')
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_best_model_selection_classification(self, model_agent):
        """Test best model selection for classification."""
        # Create mock results
        results = [
            {
                'model_name': 'Model1',
                'model': None,
                'metrics': {'f1_score': 0.8, 'accuracy': 0.85}
            },
            {
                'model_name': 'Model2',
                'model': None,
                'metrics': {'f1_score': 0.9, 'accuracy': 0.88}
            }
        ]
        
        best_result = model_agent._select_best_model(results, 'classification')
        
        assert best_result['model_name'] == 'Model2'
        assert best_result['metrics']['f1_score'] == 0.9
    
    def test_best_model_selection_regression(self, model_agent):
        """Test best model selection for regression."""
        # Create mock results
        results = [
            {
                'model_name': 'Model1',
                'model': None,
                'metrics': {'rmse': 1.5, 'r2_score': 0.8}
            },
            {
                'model_name': 'Model2',
                'model': None,
                'metrics': {'rmse': 1.2, 'r2_score': 0.85}
            }
        ]
        
        best_result = model_agent._select_best_model(results, 'regression')
        
        assert best_result['model_name'] == 'Model2'
        assert best_result['metrics']['rmse'] == 1.2
    
    def test_model_persistence(self, model_agent, sample_classification_data):
        """Test model persistence and loading."""
        # Train a model
        model_artifact = model_agent.train_and_evaluate(sample_classification_data)
        
        # Load the model
        loaded_model = model_agent.load_model(model_artifact.model_path)
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Test prediction
        predictions = model_agent.predict(loaded_model, sample_classification_data.X_test)
        assert len(predictions) == len(sample_classification_data.X_test)
    
    def test_feature_importance_extraction(self, model_agent, sample_classification_data):
        """Test feature importance extraction."""
        # Train a model that supports feature importance
        model_artifact = model_agent.train_and_evaluate(sample_classification_data)
        
        # Load the model
        model = model_agent.load_model(model_artifact.model_path)
        
        # Extract feature importance
        importance = model_agent.get_feature_importance(model, model_artifact.feature_names)
        
        # Feature importance may or may not be available depending on model type
        assert isinstance(importance, dict)
    
    def test_metrics_saving(self, model_agent, sample_classification_data):
        """Test metrics saving functionality."""
        # Train a model
        model_artifact = model_agent.train_and_evaluate(sample_classification_data)
        
        # Save metrics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metrics_path = f.name
        
        try:
            saved_path = model_agent.save_model_metrics(model_artifact.metrics, metrics_path)
            assert saved_path == metrics_path
            
            # Verify file was created
            assert Path(metrics_path).exists()
            
        finally:
            Path(metrics_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
