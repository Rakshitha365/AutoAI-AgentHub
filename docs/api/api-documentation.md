# AutoAI AgentHub - API Documentation

## Overview

The AutoAI AgentHub provides a comprehensive API for automated machine learning pipeline development. This documentation covers all available classes, methods, and their usage.

## Table of Contents

1. [Core Components](#core-components)
2. [Agent Classes](#agent-classes)
3. [Data Structures](#data-structures)
4. [Utility Functions](#utility-functions)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)
7. [Examples](#examples)

## Core Components

### Orchestrator

The main orchestrator that coordinates all agents in the AI automation pipeline.

#### Constructor

```python
Orchestrator(config_path: str)
```

**Parameters:**
- `config_path` (str): Path to configuration JSON file

**Example:**
```python
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator("src/config/config.json")
```

#### Methods

##### `run_pipeline(dataset_path, target_col=None, task_hint=None)`

Runs the complete AI automation pipeline.

**Parameters:**
- `dataset_path` (str): Path to CSV dataset file
- `target_col` (str, optional): Target column name
- `task_hint` (str, optional): Task type hint ("classification" or "regression")

**Returns:**
- `Dict[str, Any]`: Dictionary containing pipeline results

**Example:**
```python
result = orchestrator.run_pipeline(
    dataset_path="data/iris.csv",
    target_col="species",
    task_hint="classification"
)

if result['success']:
    print(f"Model accuracy: {result['model_artifact'].metrics['accuracy']}")
```

##### `get_pipeline_status()`

Gets current pipeline status.

**Returns:**
- `Dict[str, Any]`: Status information

##### `cleanup_artifacts(run_id=None)`

Cleans up artifacts for a specific run.

**Parameters:**
- `run_id` (str, optional): Specific run ID to clean up

## Agent Classes

### DataAgent

Agent responsible for data preprocessing and preparation.

#### Constructor

```python
DataAgent(config: Dict[str, Any], logger: logging.Logger)
```

#### Methods

##### `load_and_validate_data(file_path)`

Loads and validates CSV data.

**Parameters:**
- `file_path` (str): Path to CSV file

**Returns:**
- `pd.DataFrame`: Loaded and validated DataFrame

##### `preprocess_data(df, target_col=None, task_hint=None)`

Preprocesses data for machine learning.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `target_col` (str, optional): Target column name
- `task_hint` (str, optional): Task type hint

**Returns:**
- `ProcessedPayload`: Processed data structure

##### `detect_task_type(df, target_col)`

Automatically detects task type (classification/regression).

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `target_col` (str): Target column name

**Returns:**
- `str`: Task type ("classification" or "regression")

### ModelAgent

Agent responsible for model training, evaluation, and selection.

#### Constructor

```python
ModelAgent(config: Dict[str, Any], logger: logging.Logger)
```

#### Methods

##### `train_and_evaluate(payload)`

Trains and evaluates models based on processed data.

**Parameters:**
- `payload` (ProcessedPayload): Processed data payload

**Returns:**
- `ModelArtifact`: Model artifact with best model and metrics

##### `load_model(model_path)`

Loads a trained model from disk.

**Parameters:**
- `model_path` (str): Path to saved model

**Returns:**
- `Any`: Loaded model object

##### `predict(model, X)`

Makes predictions using a trained model.

**Parameters:**
- `model` (Any): Trained model
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `np.ndarray`: Predictions array

##### `get_feature_importance(model, feature_names)`

Gets feature importance from model.

**Parameters:**
- `model` (Any): Trained model
- `feature_names` (List[str]): List of feature names

**Returns:**
- `Dict[str, float]`: Feature importance scores

### DeployAgent

Agent responsible for model deployment and interface generation.

#### Constructor

```python
DeployAgent(config: Dict[str, Any], logger: logging.Logger)
```

#### Methods

##### `generate_streamlit_app(model_artifact, run_id)`

Generates a Streamlit application for model deployment.

**Parameters:**
- `model_artifact` (ModelArtifact): Model artifact
- `run_id` (str): Run identifier

**Returns:**
- `str`: Path to generated Streamlit app

##### `generate_api(model_artifact, run_id)`

Generates a FastAPI application for model deployment.

**Parameters:**
- `model_artifact` (ModelArtifact): Model artifact
- `run_id` (str): Run identifier

**Returns:**
- `str`: Path to generated API

##### `launch_streamlit_app(app_path)`

Launches a Streamlit application.

**Parameters:**
- `app_path` (str): Path to Streamlit app

**Returns:**
- `bool`: Success status

## Data Structures

### ProcessedPayload

Data structure containing processed dataset splits and metadata.

```python
@dataclass
class ProcessedPayload:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    target_column: str
    task_type: str
    meta: Dict[str, Any]
```

### ModelArtifact

Data structure containing trained model and evaluation metrics.

```python
@dataclass
class ModelArtifact:
    model: Any
    model_type: str
    task_type: str
    metrics: Dict[str, float]
    feature_names: List[str]
    target_column: str
    timestamp: str
    model_path: str
```

### DeploymentInfo

Data structure containing deployment information.

```python
@dataclass
class DeploymentInfo:
    streamlit_app_path: str
    api_path: str
    model_path: str
    metrics_path: str
    timestamp: str
```

## Utility Functions

### Validation Functions

#### `validate_dataframe(df, min_rows=1, min_cols=2)`

Validates DataFrame structure and content.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `min_rows` (int): Minimum number of rows required
- `min_cols` (int): Minimum number of columns required

**Returns:**
- `Tuple[bool, str]`: Validation result and message

#### `calculate_data_quality_score(df)`

Calculates data quality score.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze

**Returns:**
- `float`: Quality score (0-1)

#### `detect_column_types(df)`

Detects column types (numerical, categorical, datetime).

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze

**Returns:**
- `Dict[str, str]`: Column type mapping

### File Operations

#### `create_directory_structure(base_path)`

Creates standard directory structure.

**Parameters:**
- `base_path` (str): Base directory path

#### `save_json(data, file_path)`

Saves data to JSON file.

**Parameters:**
- `data` (Any): Data to save
- `file_path` (str): Output file path

#### `load_json(file_path)`

Loads data from JSON file.

**Parameters:**
- `file_path` (str): Input file path

**Returns:**
- `Any`: Loaded data

## Configuration

### Configuration File Structure

The configuration is stored in `src/config/config.json`:

```json
{
  "max_file_size_mb": 10,
  "default_test_size": 0.2,
  "random_state": 42,
  "max_categorical_cardinality": 50,
  "streamlit_port": 8501,
  "log_level": "INFO",
  "artifact_dir": "artifacts",
  "models_dir": "artifacts/models",
  "logs_dir": "artifacts/logs",
  "reports_dir": "artifacts/reports",
  "model_trials": {
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
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|--------------|
| `max_file_size_mb` | int | 10 | Maximum file size in MB |
| `default_test_size` | float | 0.2 | Default test split ratio |
| `random_state` | int | 42 | Random seed for reproducibility |
| `max_categorical_cardinality` | int | 50 | Max categories for encoding |
| `streamlit_port` | int | 8501 | Streamlit server port |
| `log_level` | str | "INFO" | Logging level |
| `artifact_dir` | str | "artifacts" | Base artifacts directory |
| `models_dir` | str | "artifacts/models" | Models directory |
| `logs_dir` | str | "artifacts/logs" | Logs directory |
| `reports_dir` | str | "artifacts/reports" | Reports directory |

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
Raised when required files are not found.

#### `ValueError`
Raised for invalid data or parameters.

#### `RuntimeError`
Raised for pipeline execution errors.

### Error Handling Best Practices

```python
try:
    result = orchestrator.run_pipeline("data.csv")
    if result['success']:
        print("Pipeline completed successfully")
    else:
        print(f"Pipeline failed: {result['error']}")
except FileNotFoundError:
    print("Dataset file not found")
except ValueError as e:
    print(f"Invalid data: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

### Basic Usage

```python
from src.core.orchestrator import Orchestrator

# Initialize orchestrator
orchestrator = Orchestrator("src/config/config.json")

# Run complete pipeline
result = orchestrator.run_pipeline(
    dataset_path="data/iris.csv",
    target_col="species"
)

# Check results
if result['success']:
    model = result['model_artifact']
    print(f"Best model: {model.model_type}")
    print(f"Accuracy: {model.metrics['accuracy']:.4f}")
```

### Custom Configuration

```python
import json
from src.core.orchestrator import Orchestrator

# Load custom configuration
with open("custom_config.json", "r") as f:
    config = json.load(f)

# Update configuration
config["random_state"] = 123
config["default_test_size"] = 0.3

# Save updated configuration
with open("src/config/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Use with custom config
orchestrator = Orchestrator("src/config/config.json")
```

### Individual Agent Usage

```python
from src.agents.data_agent import DataAgent
from src.agents.model_agent import ModelAgent
import logging

# Setup logging
logger = logging.getLogger("example")

# Initialize agents
config = {"random_state": 42, "default_test_size": 0.2}
data_agent = DataAgent(config, logger)
model_agent = ModelAgent(config, logger)

# Process data
df = data_agent.load_and_validate_data("data.csv")
payload = data_agent.preprocess_data(df, target_col="target")

# Train model
model_artifact = model_agent.train_and_evaluate(payload)
print(f"Model trained: {model_artifact.model_type}")
```

### Batch Processing

```python
import os
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator("src/config/config.json")

# Process multiple datasets
datasets = ["iris.csv", "wine.csv", "breast_cancer.csv"]
results = []

for dataset in datasets:
    if os.path.exists(f"data/{dataset}"):
        result = orchestrator.run_pipeline(f"data/{dataset}")
        results.append({
            "dataset": dataset,
            "success": result['success'],
            "accuracy": result['model_artifact'].metrics.get('accuracy', 0)
        })

# Summary
for result in results:
    print(f"{result['dataset']}: {result['accuracy']:.4f}")
```

## Performance Considerations

### Memory Usage
- Large datasets (>1GB) may require chunked processing
- Consider using `dtype` optimization for pandas DataFrames
- Monitor memory usage with `psutil` for large-scale operations

### Processing Speed
- Use parallel processing for multiple model training
- Consider GPU acceleration for deep learning models
- Profile bottlenecks with `cProfile` or `line_profiler`

### Best Practices
- Use appropriate data types to reduce memory footprint
- Implement caching for repeated operations
- Use incremental learning for large datasets
- Monitor resource usage during long-running processes

---

*This documentation is automatically generated and updated with each release of AutoAI AgentHub.*
