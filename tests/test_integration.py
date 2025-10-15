import pytest
import pandas as pd
import numpy as np
import tempfile
import logging
from pathlib import Path
import sys
import json
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from src.agents.data_agent import DataAgent
from src.agents.model_agent import ModelAgent
from src.agents.deploy_agent import DeployAgent


class TestIntegration:
    """Integration tests for the complete AutoAI framework."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "artifact_dir": "test_artifacts",
            "models_dir": "test_artifacts/models",
            "logs_dir": "test_artifacts/logs",
            "reports_dir": "test_artifacts/reports",
            "max_file_size_mb": 10,
            "default_test_size": 0.2,
            "random_state": 42,
            "max_categorical_cardinality": 50,
            "streamlit_port": 8501,
            "model_trials": {
                "classification": ["LogisticRegression", "DecisionTreeClassifier"],
                "regression": ["LinearRegression", "Ridge"]
            }
        }
    
    @pytest.fixture
    def config_file(self, sample_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            return f.name
    
    @pytest.fixture
    def sample_classification_data(self):
        """Sample classification dataset."""
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_regression_data(self):
        """Sample regression dataset."""
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['X', 'Y', 'Z'], 100),
            'target': np.random.normal(0, 1, 100)
        }
        return pd.DataFrame(data)
    
    def test_end_to_end_classification_pipeline(self, config_file, sample_classification_data):
        """Test complete classification pipeline."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_classification_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Initialize orchestrator
            orchestrator = Orchestrator(config_file)
            
            # Run complete pipeline
            result = orchestrator.run_pipeline(
                dataset_path=csv_path,
                target_col='target'
            )
            
            # Validate results
            assert result['success'] is True
            assert 'run_id' in result
            assert 'model_artifact' in result
            assert 'deployment_info' in result
            
            # Validate model artifact
            model_artifact = result['model_artifact']
            assert model_artifact.task_type == 'classification'
            assert model_artifact.model_type in ['LogisticRegression', 'DecisionTreeClassifier']
            assert 'accuracy' in model_artifact.metrics
            assert 'f1_score' in model_artifact.metrics
            
            # Validate deployment info
            deployment_info = result['deployment_info']
            assert deployment_info.status == 'ready'
            assert Path(deployment_info.app_path).exists()
            
            # Validate artifacts
            assert 'model' in result['artifacts']
            assert 'preprocessor' in result['artifacts']
            assert 'metrics' in result['artifacts']
            
        finally:
            # Cleanup
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def test_end_to_end_regression_pipeline(self, config_file, sample_regression_data):
        """Test complete regression pipeline."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_regression_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Initialize orchestrator
            orchestrator = Orchestrator(config_file)
            
            # Run complete pipeline
            result = orchestrator.run_pipeline(
                dataset_path=csv_path,
                target_col='target'
            )
            
            # Validate results
            assert result['success'] is True
            assert 'run_id' in result
            assert 'model_artifact' in result
            assert 'deployment_info' in result
            
            # Validate model artifact
            model_artifact = result['model_artifact']
            assert model_artifact.task_type == 'regression'
            assert model_artifact.model_type in ['LinearRegression', 'Ridge']
            assert 'mse' in model_artifact.metrics
            assert 'r2_score' in model_artifact.metrics
            
            # Validate deployment info
            deployment_info = result['deployment_info']
            assert deployment_info.status == 'ready'
            assert Path(deployment_info.app_path).exists()
            
        finally:
            # Cleanup
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def test_auto_target_detection(self, config_file):
        """Test automatic target column detection."""
        # Create dataset with clear target and mixed features
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'category': np.random.choice(['A', 'B', 'C'], 50),
            'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 50)
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            orchestrator = Orchestrator(config_file)
            result = orchestrator.run_pipeline(dataset_path=csv_path)
            
            assert result['success'] is True
            assert result['model_artifact'].target_column == 'category'  # Auto-detected target
            assert result['model_artifact'].task_type == 'classification'
            
        finally:
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def test_error_handling_invalid_file(self, config_file):
        """Test error handling for invalid files."""
        orchestrator = Orchestrator(config_file)
        
        result = orchestrator.run_pipeline(dataset_path="nonexistent_file.csv")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_error_handling_empty_dataset(self, config_file):
        """Test error handling for empty datasets."""
        empty_df = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            orchestrator = Orchestrator(config_file)
            result = orchestrator.run_pipeline(dataset_path=csv_path)
            
            assert result['success'] is False
            assert 'error' in result
            
        finally:
            os.unlink(csv_path)
    
    def test_streamlit_app_generation(self, config_file, sample_classification_data):
        """Test Streamlit app generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_classification_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            orchestrator = Orchestrator(config_file)
            result = orchestrator.run_pipeline(
                dataset_path=csv_path,
                target_col='target'
            )
            
            assert result['success'] is True
            
            # Check if Streamlit app was generated
            app_path = result['deployment_info'].app_path
            assert Path(app_path).exists()
            
            # Check app content
            with open(app_path, 'r', encoding='utf-8') as f:
                app_content = f.read()
                assert 'streamlit' in app_content.lower()
                assert 'AutoAI' in app_content
                assert 'prediction' in app_content.lower()
            
        finally:
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def test_model_persistence_and_loading(self, config_file, sample_classification_data):
        """Test model persistence and loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_classification_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            orchestrator = Orchestrator(config_file)
            result = orchestrator.run_pipeline(
                dataset_path=csv_path,
                target_col='target'
            )
            
            assert result['success'] is True
            
            # Load the model
            model_artifact = result['model_artifact']
            
            # Load config from file
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            model_agent = ModelAgent(config, logging.getLogger('test'))
            
            loaded_model = model_agent.load_model(model_artifact.model_path)
            assert loaded_model is not None
            
            # Test prediction using preprocessed data
            # Create simple test data that matches the model's expected features
            # The model was trained on preprocessed data, so we need to create compatible test data
            import numpy as np
            test_data = np.array([[0.5, -1.4, 0, 1, 0],  # feature1, feature2, feature3_A, feature3_B, feature3_C
                                 [-0.1, -0.4, 0, 1, 0],
                                 [0.6, -0.3, 1, 0, 0],
                                 [1.5, -0.8, 1, 0, 0],
                                 [-0.2, -0.2, 1, 0, 0]])
            test_df = pd.DataFrame(test_data, columns=['feature1', 'feature2', 'feature3_A', 'feature3_B', 'feature3_C'])
            predictions = model_agent.predict(loaded_model, test_df)
            assert len(predictions) == 5
            
        finally:
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def test_artifacts_generation(self, config_file, sample_classification_data):
        """Test that all artifacts are properly generated."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_classification_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            orchestrator = Orchestrator(config_file)
            result = orchestrator.run_pipeline(
                dataset_path=csv_path,
                target_col='target'
            )
            
            assert result['success'] is True
            
            artifacts = result['artifacts']
            
            # Check that all expected artifacts exist
            expected_artifacts = ['model', 'preprocessor', 'metrics', 'streamlit_app', 'summary']
            
            for artifact_type in expected_artifacts:
                assert artifact_type in artifacts
                assert Path(artifacts[artifact_type]).exists()
            
        finally:
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def test_performance_benchmarks(self, config_file, sample_classification_data):
        """Test that performance meets basic benchmarks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_classification_data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            import time
            start_time = time.time()
            
            orchestrator = Orchestrator(config_file)
            result = orchestrator.run_pipeline(
                dataset_path=csv_path,
                target_col='target'
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert result['success'] is True
            assert execution_time < 30  # Should complete within 30 seconds
            
            # Check model performance
            model_artifact = result['model_artifact']
            if model_artifact.task_type == 'classification':
                assert model_artifact.metrics['accuracy'] >= 0.5  # At least 50% accuracy
            
        finally:
            os.unlink(csv_path)
            self._cleanup_test_artifacts()
    
    def _cleanup_test_artifacts(self):
        """Clean up test artifacts with Windows-compatible approach."""
        import shutil
        import time
        import gc
        
        if Path('test_artifacts').exists():
            # Force garbage collection to release file handles
            gc.collect()
            time.sleep(0.5)
            
            try:
                shutil.rmtree('test_artifacts')
            except (PermissionError, OSError):
                # Windows-specific cleanup with multiple strategies
                try:
                    # Strategy 1: Use os.walk with error handling
                    for root, dirs, files in os.walk('test_artifacts', topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.unlink(file_path)
                            except (PermissionError, OSError):
                                # Try to change attributes and retry
                                try:
                                    os.chmod(file_path, 0o777)
                                    os.unlink(file_path)
                                except:
                                    pass
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except (PermissionError, OSError):
                                pass
                    os.rmdir('test_artifacts')
                except (PermissionError, OSError):
                    # Strategy 2: Use subprocess to force delete (Windows only)
                    try:
                        import subprocess
                        subprocess.run(['rmdir', '/s', '/q', 'test_artifacts'], 
                                     shell=True, capture_output=True)
                    except:
                        # Strategy 3: Just leave it - it's a test artifact
                        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
