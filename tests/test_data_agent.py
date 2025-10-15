import pytest
import pandas as pd
import numpy as np
import tempfile
import logging
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.data_agent import DataAgent


class TestDataAgent:
    """Test cases for Data Agent."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'max_file_size_mb': 10,
            'default_test_size': 0.2,
            'random_state': 42,
            'max_categorical_cardinality': 50
        }
    
    @pytest.fixture
    def sample_logger(self):
        """Sample logger for testing."""
        return logging.getLogger('test')
    
    @pytest.fixture
    def data_agent(self, sample_config, sample_logger):
        """Data Agent instance for testing."""
        return DataAgent(sample_config, sample_logger)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        }
        return pd.DataFrame(data)
    
    def test_data_agent_initialization(self, data_agent):
        """Test Data Agent initialization."""
        assert data_agent is not None
        assert data_agent.max_file_size_mb == 10
        assert data_agent.test_size == 0.2
        assert data_agent.random_state == 42
    
    def test_load_dataframe(self, data_agent, sample_dataframe):
        """Test loading DataFrame."""
        loaded_df = data_agent.load(sample_dataframe)
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 5
        assert len(loaded_df.columns) == 4
    
    def test_load_csv_file(self, data_agent, sample_dataframe):
        """Test loading CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = data_agent.load(temp_path)
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) == 5
            assert len(loaded_df.columns) == 4
        finally:
            # Close the file handle before trying to delete
            try:
                os.unlink(temp_path)
            except PermissionError:
                # On Windows, sometimes files are still locked
                pass
    
    def test_analyze_dataframe(self, data_agent, sample_dataframe):
        """Test DataFrame analysis."""
        metadata = data_agent.analyze(sample_dataframe)
        
        assert 'n_samples' in metadata
        assert 'n_features' in metadata
        assert 'columns' in metadata
        assert 'dtypes' in metadata
        assert 'missing_values' in metadata
        assert 'categorical_columns' in metadata
        assert 'numerical_columns' in metadata
        assert 'target_candidates' in metadata
        
        assert metadata['n_samples'] == 5
        assert metadata['n_features'] == 4
        assert len(metadata['target_candidates']) > 0
    
    def test_preprocess_dataframe(self, data_agent, sample_dataframe):
        """Test DataFrame preprocessing."""
        processed_payload = data_agent.preprocess(sample_dataframe, 'target')
        
        assert processed_payload is not None
        assert len(processed_payload.X_train) > 0
        assert len(processed_payload.X_test) > 0
        assert len(processed_payload.y_train) > 0
        assert len(processed_payload.y_test) > 0
        assert processed_payload.meta['task_type'] in ['classification', 'regression']
    
    def test_validate_data_quality(self, data_agent, sample_dataframe):
        """Test data quality validation."""
        quality_report = data_agent.validate_data_quality(sample_dataframe)
        
        assert 'total_rows' in quality_report
        assert 'total_columns' in quality_report
        assert 'missing_values_count' in quality_report
        assert 'quality_score' in quality_report
        
        assert quality_report['total_rows'] == 5
        assert quality_report['total_columns'] == 4
        assert quality_report['quality_score'] >= 0
    
    def test_missing_values_handling(self, data_agent):
        """Test handling of missing values."""
        # Create DataFrame with missing values
        data_with_missing = {
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [0.1, np.nan, 0.3, 0.4, 0.5],
            'category': ['A', 'B', None, 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        }
        df_with_missing = pd.DataFrame(data_with_missing)
        
        # Preprocess
        processed_payload = data_agent.preprocess(df_with_missing, 'target')
        
        # Check that missing values are handled
        assert not processed_payload.X_train.isnull().any().any()
        assert not processed_payload.X_test.isnull().any().any()
    
    def test_invalid_file_handling(self, data_agent):
        """Test handling of invalid files."""
        with pytest.raises(FileNotFoundError):
            data_agent.load("nonexistent_file.csv")
    
    def test_empty_dataframe_handling(self, data_agent):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            data_agent.load(empty_df)
    
    def test_insufficient_columns_handling(self, data_agent):
        """Test handling of DataFrame with insufficient columns."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            data_agent.load(single_col_df)
    
    def test_target_column_not_found(self, data_agent, sample_dataframe):
        """Test handling when target column is not found."""
        with pytest.raises(ValueError):
            data_agent.preprocess(sample_dataframe, 'nonexistent_target')


if __name__ == "__main__":
    pytest.main([__file__])
