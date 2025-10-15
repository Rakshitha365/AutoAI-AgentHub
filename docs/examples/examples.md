# AutoAI AgentHub - Examples & Tutorials

## Getting Started Examples

Welcome to AutoAI AgentHub! This section provides comprehensive examples and tutorials to help you get started with automated machine learning.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Classification Examples](#classification-examples)
3. [Regression Examples](#regression-examples)
4. [Advanced Examples](#advanced-examples)
5. [Custom Configuration Examples](#custom-configuration-examples)
6. [API Usage Examples](#api-usage-examples)
7. [Troubleshooting Examples](#troubleshooting-examples)

## Quick Start Examples

### Example 1: Basic Classification

**Scenario**: Classify iris flower species based on measurements.

**Data**: Iris dataset (150 samples, 4 features, 3 classes)

**Steps**:

1. **Prepare Data**
   ```python
   # Sample iris data structure
   import pandas as pd
   
   data = {
       'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
       'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
       'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
       'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
       'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
   }
   df = pd.DataFrame(data)
   df.to_csv('iris_sample.csv', index=False)
   ```

2. **Run Pipeline**
   ```bash
   python main.py --dataset iris_sample.csv --target species
   ```

3. **Expected Output**
   ```
   🤖 AutoAI AgentHub - Starting Pipeline
   ==================================================
   📁 Dataset: iris_sample.csv
   🎯 Target: species
   🔧 Task: classification
   
   🔄 Processing Data...
   ✅ Data loaded: 150 rows, 5 columns
   ✅ Missing values handled
   ✅ Categorical encoding applied
   ✅ Data split: 120 train, 30 test
   
   🎯 Training Models...
   ✅ Logistic Regression: accuracy=0.97
   ✅ Decision Tree: accuracy=0.93
   ✅ Random Forest: accuracy=0.97
   
   🏆 Best Model: Random Forest (accuracy=0.97)
   ```

### Example 2: Basic Regression

**Scenario**: Predict house prices based on features.

**Data**: California housing dataset (20,640 samples, 8 features)

**Steps**:

1. **Run Pipeline**
   ```bash
   python main.py --dataset data/california_housing.csv --target MEDV --task regression
   ```

2. **Expected Results**
   ```
   🎯 Training Models...
   ✅ Linear Regression: R²=0.61, RMSE=0.73
   ✅ Ridge Regression: R²=0.61, RMSE=0.73
   ✅ Random Forest: R²=0.81, RMSE=0.52
   
   🏆 Best Model: Random Forest (R²=0.81)
   ```

### Example 3: Web Interface Usage

**Steps**:

1. **Launch Web Interface**
   ```bash
   python main.py --web
   ```

2. **Navigate to Browser**
   - Open `http://localhost:8501`
   - Upload your CSV file
   - Configure settings
   - Click "Generate AI Model"

3. **Interactive Results**
   - View performance metrics
   - Test predictions
   - Download model artifacts

## Classification Examples

### Example 1: Customer Churn Prediction

**Scenario**: Predict customer churn for a telecom company.

**Data Structure**:
```csv
customer_id,age,monthly_charges,contract_length,churn
1,25,29.85,12,No
2,56,56.05,24,Yes
3,45,42.30,12,No
```

**Implementation**:
```python
from src.core.orchestrator import Orchestrator

# Initialize orchestrator
orchestrator = Orchestrator("src/config/config.json")

# Run churn prediction pipeline
result = orchestrator.run_pipeline(
    dataset_path="data/customer_churn.csv",
    target_col="churn",
    task_hint="classification"
)

if result['success']:
    model = result['model_artifact']
    print(f"Churn Prediction Model:")
    print(f"- Model Type: {model.model_type}")
    print(f"- Accuracy: {model.metrics['accuracy']:.4f}")
    print(f"- F1-Score: {model.metrics['f1_score']:.4f}")
    
    # Test prediction
    test_data = [[35, 45.50, 24]]  # age, monthly_charges, contract_length
    prediction = model.model.predict(test_data)
    print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
```

### Example 2: Email Spam Detection

**Scenario**: Classify emails as spam or not spam.

**Data Features**:
- Word frequency counts
- Email length
- Number of links
- Number of images

**Implementation**:
```python
import pandas as pd
from src.agents.data_agent import DataAgent
from src.agents.model_agent import ModelAgent
import logging

# Setup
logger = logging.getLogger("spam_detection")
config = {"random_state": 42, "default_test_size": 0.2}

# Initialize agents
data_agent = DataAgent(config, logger)
model_agent = ModelAgent(config, logger)

# Load and process data
df = data_agent.load_and_validate_data("data/spam_emails.csv")
payload = data_agent.preprocess_data(df, target_col="is_spam")

# Train model
model_artifact = model_agent.train_and_evaluate(payload)

# Analyze results
print("Spam Detection Results:")
print(f"Best Model: {model_artifact.model_type}")
print(f"Accuracy: {model_artifact.metrics['accuracy']:.4f}")
print(f"Precision: {model_artifact.metrics['precision']:.4f}")
print(f"Recall: {model_artifact.metrics['recall']:.4f}")

# Feature importance
importance = model_agent.get_feature_importance(
    model_artifact.model, 
    model_artifact.feature_names
)
print("\nTop Features:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"- {feature}: {score:.4f}")
```

### Example 3: Multi-Class Classification

**Scenario**: Classify news articles into categories (sports, politics, technology, etc.).

**Implementation**:
```python
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator("src/config/config.json")

# Run multi-class classification
result = orchestrator.run_pipeline(
    dataset_path="data/news_articles.csv",
    target_col="category",
    task_hint="classification"
)

if result['success']:
    model = result['model_artifact']
    
    # Get class predictions
    test_text = "The team won the championship game yesterday"
    prediction = model.model.predict([test_text])
    
    # Get prediction probabilities
    probabilities = model.model.predict_proba([test_text])
    classes = model.model.classes_
    
    print("News Classification Results:")
    for i, prob in enumerate(probabilities[0]):
        print(f"- {classes[i]}: {prob:.4f}")
```

## Regression Examples

### Example 1: Sales Forecasting

**Scenario**: Predict monthly sales based on historical data and features.

**Data Features**:
- Previous month sales
- Marketing spend
- Seasonality indicators
- Economic indicators

**Implementation**:
```python
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator("src/config/config.json")

# Run sales forecasting pipeline
result = orchestrator.run_pipeline(
    dataset_path="data/sales_data.csv",
    target_col="monthly_sales",
    task_hint="regression"
)

if result['success']:
    model = result['model_artifact']
    
    print("Sales Forecasting Model:")
    print(f"- Model Type: {model.model_type}")
    print(f"- R² Score: {model.metrics['r2_score']:.4f}")
    print(f"- RMSE: {model.metrics['rmse']:.2f}")
    print(f"- MAE: {model.metrics['mae']:.2f}")
    
    # Make future prediction
    future_features = [[50000, 10000, 1, 0.02]]  # prev_sales, marketing, season, econ_indicator
    prediction = model.model.predict(future_features)
    print(f"Predicted Sales: ${prediction[0]:,.2f}")
```

### Example 2: Stock Price Prediction

**Scenario**: Predict stock prices based on technical indicators.

**Implementation**:
```python
import pandas as pd
import numpy as np
from src.agents.data_agent import DataAgent
from src.agents.model_agent import ModelAgent

# Create sample stock data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = {
    'date': dates,
    'open': np.random.normal(100, 5, 100),
    'high': np.random.normal(105, 5, 100),
    'low': np.random.normal(95, 5, 100),
    'close': np.random.normal(100, 5, 100),
    'volume': np.random.randint(1000, 10000, 100),
    'rsi': np.random.uniform(20, 80, 100),
    'macd': np.random.normal(0, 2, 100)
}

# Add target (next day close price)
df = pd.DataFrame(data)
df['target'] = df['close'].shift(-1)
df = df.dropna()

# Save data
df.to_csv('stock_data.csv', index=False)

# Run prediction pipeline
from src.core.orchestrator import Orchestrator
orchestrator = Orchestrator("src/config/config.json")

result = orchestrator.run_pipeline(
    dataset_path="stock_data.csv",
    target_col="target",
    task_hint="regression"
)

if result['success']:
    model = result['model_artifact']
    print(f"Stock Prediction Model: {model.model_type}")
    print(f"R² Score: {model.metrics['r2_score']:.4f}")
```

### Example 3: Energy Consumption Prediction

**Scenario**: Predict energy consumption for buildings.

**Implementation**:
```python
from src.core.orchestrator import Orchestrator

# Custom configuration for energy prediction
config = {
    "max_file_size_mb": 20,
    "default_test_size": 0.2,
    "random_state": 42,
    "model_trials": {
        "regression": [
            "LinearRegression",
            "Ridge",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
        ]
    }
}

# Save custom config
import json
with open("energy_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Run energy prediction
orchestrator = Orchestrator("energy_config.json")
result = orchestrator.run_pipeline(
    dataset_path="data/energy_consumption.csv",
    target_col="consumption_kwh"
)

if result['success']:
    model = result['model_artifact']
    print("Energy Consumption Prediction:")
    print(f"- Model: {model.model_type}")
    print(f"- R² Score: {model.metrics['r2_score']:.4f}")
    print(f"- RMSE: {model.metrics['rmse']:.2f} kWh")
```

## Advanced Examples

### Example 1: Custom Model Integration

**Scenario**: Add a custom machine learning model to the framework.

**Implementation**:
```python
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class CustomRegressor(BaseEstimator, RegressorMixin):
    """Custom regression model for demonstration."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train the custom model."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias

# Integrate custom model
from src.agents.model_agent import ModelAgent

class ExtendedModelAgent(ModelAgent):
    """Extended model agent with custom model support."""
    
    def _get_models(self, task_type):
        """Get models including custom models."""
        models = super()._get_models(task_type)
        
        if task_type == "regression":
            models["CustomRegressor"] = CustomRegressor()
        
        return models

# Usage
config = {"random_state": 42}
logger = logging.getLogger("custom_model")
agent = ExtendedModelAgent(config, logger)

# Train with custom model
payload = data_agent.preprocess_data(df, target_col="target")
model_artifact = agent.train_and_evaluate(payload)
```

### Example 2: Batch Processing Multiple Datasets

**Scenario**: Process multiple datasets in batch for comparison.

**Implementation**:
```python
import os
import pandas as pd
from src.core.orchestrator import Orchestrator

def batch_process_datasets(dataset_paths, output_file="batch_results.csv"):
    """Process multiple datasets and compare results."""
    
    orchestrator = Orchestrator("src/config/config.json")
    results = []
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"Processing {dataset_path}...")
            
            try:
                result = orchestrator.run_pipeline(dataset_path)
                
                if result['success']:
                    model = result['model_artifact']
                    results.append({
                        'dataset': os.path.basename(dataset_path),
                        'task_type': model.task_type,
                        'model_type': model.model_type,
                        'accuracy': model.metrics.get('accuracy', 0),
                        'f1_score': model.metrics.get('f1_score', 0),
                        'r2_score': model.metrics.get('r2_score', 0),
                        'rmse': model.metrics.get('rmse', 0),
                        'processing_time': result.get('processing_time', 0)
                    })
                else:
                    print(f"Failed to process {dataset_path}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"Error processing {dataset_path}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nBatch Processing Summary:")
    print(results_df.to_string(index=False))
    
    return results_df

# Usage
datasets = [
    "data/iris.csv",
    "data/wine.csv", 
    "data/breast_cancer.csv",
    "data/california_housing.csv"
]

results = batch_process_datasets(datasets)
```

### Example 3: Real-time Prediction Service

**Scenario**: Create a real-time prediction service using the generated API.

**Implementation**:
```python
import requests
import json
import time
from threading import Thread

class PredictionService:
    """Real-time prediction service."""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.is_running = False
    
    def start_service(self):
        """Start the prediction service."""
        self.is_running = True
        print("Prediction service started...")
        
        while self.is_running:
            try:
                # Health check
                response = requests.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    print("Service is healthy")
                else:
                    print("Service health check failed")
                    
            except requests.exceptions.RequestException as e:
                print(f"Service unavailable: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def make_prediction(self, features):
        """Make a prediction request."""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={"features": features}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def stop_service(self):
        """Stop the prediction service."""
        self.is_running = False
        print("Prediction service stopped")

# Usage
service = PredictionService()

# Start service in background
service_thread = Thread(target=service.start_service)
service_thread.daemon = True
service_thread.start()

# Make predictions
test_features = [5.1, 3.5, 1.4, 0.2]
result = service.make_prediction(test_features)
print(f"Prediction: {result}")

# Stop service
service.stop_service()
```

## Custom Configuration Examples

### Example 1: High-Performance Configuration

**Scenario**: Optimize configuration for large datasets and high performance.

**Configuration**:
```json
{
  "max_file_size_mb": 100,
  "default_test_size": 0.2,
  "random_state": 42,
  "max_categorical_cardinality": 1000,
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
      "RandomForestClassifier",
      "GradientBoostingClassifier",
      "SVC"
    ],
    "regression": [
      "LinearRegression",
      "Ridge",
      "Lasso",
      "RandomForestRegressor",
      "GradientBoostingRegressor",
      "SVR"
    ]
  },
  "performance": {
    "n_jobs": -1,
    "verbose": 1,
    "early_stopping": true,
    "cross_validation_folds": 5
  }
}
```

### Example 2: Research Configuration

**Scenario**: Configuration optimized for research and experimentation.

**Configuration**:
```json
{
  "max_file_size_mb": 50,
  "default_test_size": 0.3,
  "random_state": 42,
  "max_categorical_cardinality": 100,
  "streamlit_port": 8502,
  "log_level": "DEBUG",
  "artifact_dir": "research_artifacts",
  "models_dir": "research_artifacts/models",
  "logs_dir": "research_artifacts/logs",
  "reports_dir": "research_artifacts/reports",
  "model_trials": {
    "classification": [
      "LogisticRegression",
      "DecisionTreeClassifier",
      "RandomForestClassifier",
      "GradientBoostingClassifier",
      "AdaBoostClassifier",
      "ExtraTreesClassifier"
    ],
    "regression": [
      "LinearRegression",
      "Ridge",
      "Lasso",
      "ElasticNet",
      "RandomForestRegressor",
      "GradientBoostingRegressor",
      "AdaBoostRegressor",
      "ExtraTreesRegressor"
    ]
  },
  "research": {
    "save_intermediate_results": true,
    "detailed_logging": true,
    "experiment_tracking": true,
    "cross_validation_folds": 10,
    "hyperparameter_tuning": true
  }
}
```

### Example 3: Production Configuration

**Scenario**: Configuration optimized for production deployment.

**Configuration**:
```json
{
  "max_file_size_mb": 20,
  "default_test_size": 0.2,
  "random_state": 42,
  "max_categorical_cardinality": 50,
  "streamlit_port": 8501,
  "log_level": "WARNING",
  "artifact_dir": "/var/lib/autoai/artifacts",
  "models_dir": "/var/lib/autoai/models",
  "logs_dir": "/var/log/autoai",
  "reports_dir": "/var/lib/autoai/reports",
  "model_trials": {
    "classification": [
      "LogisticRegression",
      "RandomForestClassifier"
    ],
    "regression": [
      "LinearRegression",
      "RandomForestRegressor"
    ]
  },
  "production": {
    "enable_monitoring": true,
    "enable_metrics": true,
    "enable_health_checks": true,
    "max_concurrent_requests": 100,
    "request_timeout": 30,
    "enable_caching": true,
    "cache_ttl": 3600
  }
}
```

## API Usage Examples

### Example 1: Basic API Integration

**Scenario**: Integrate AutoAI AgentHub API into an existing application.

**Implementation**:
```python
import requests
import json

class AutoAIClient:
    """Client for AutoAI AgentHub API."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def train_model(self, dataset_path, target_col=None, task_hint=None):
        """Train a model via API."""
        data = {
            "dataset_path": dataset_path,
            "target_col": target_col,
            "task_hint": task_hint
        }
        
        response = self.session.post(
            f"{self.base_url}/train",
            json=data
        )
        
        return response.json()
    
    def predict(self, model_id, features):
        """Make predictions via API."""
        data = {
            "model_id": model_id,
            "features": features
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=data
        )
        
        return response.json()
    
    def get_model_info(self, model_id):
        """Get model information."""
        response = self.session.get(f"{self.base_url}/models/{model_id}")
        return response.json()
    
    def list_models(self):
        """List all available models."""
        response = self.session.get(f"{self.base_url}/models")
        return response.json()

# Usage
client = AutoAIClient()

# Train a model
result = client.train_model("data/iris.csv", target_col="species")
model_id = result["model_id"]

# Make predictions
prediction = client.predict(model_id, [5.1, 3.5, 1.4, 0.2])
print(f"Prediction: {prediction}")

# Get model information
info = client.get_model_info(model_id)
print(f"Model accuracy: {info['metrics']['accuracy']}")
```

### Example 2: RESTful API Integration

**Scenario**: Create a RESTful service that uses AutoAI AgentHub.

**Implementation**:
```python
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# AutoAI AgentHub configuration
AUTOAI_BASE_URL = os.getenv("AUTOAI_BASE_URL", "http://localhost:8000")

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        
        # Validate input
        if 'features' not in data:
            return jsonify({"error": "Features required"}), 400
        
        # Make prediction request to AutoAI
        response = requests.post(
            f"{AUTOAI_BASE_URL}/predict",
            json=data
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Prediction failed"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Training endpoint."""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['dataset_path', 'target_col']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} required"}), 400
        
        # Make training request to AutoAI
        response = requests.post(
            f"{AUTOAI_BASE_URL}/train",
            json=data
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Training failed"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        response = requests.get(f"{AUTOAI_BASE_URL}/health")
        if response.status_code == 200:
            return jsonify({"status": "healthy"})
        else:
            return jsonify({"status": "unhealthy"}), 500
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## Troubleshooting Examples

### Example 1: Data Quality Issues

**Problem**: Dataset has too many missing values.

**Solution**:
```python
import pandas as pd
from src.utils.validation import validate_dataframe, calculate_data_quality_score

# Load problematic dataset
df = pd.read_csv("problematic_data.csv")

# Check data quality
is_valid, message = validate_dataframe(df)
quality_score = calculate_data_quality_score(df)

print(f"Data valid: {is_valid}")
print(f"Quality score: {quality_score:.4f}")
print(f"Message: {message}")

# Clean data
if quality_score < 0.7:
    print("Data quality is poor. Cleaning data...")
    
    # Remove columns with >50% missing values
    threshold = 0.5
    df_cleaned = df.dropna(thresh=int(len(df) * threshold), axis=1)
    
    # Fill remaining missing values
    df_cleaned = df_cleaned.fillna(df_cleaned.median())
    
    # Save cleaned data
    df_cleaned.to_csv("cleaned_data.csv", index=False)
    print("Cleaned data saved to cleaned_data.csv")
```

### Example 2: Memory Issues

**Problem**: Large dataset causes memory errors.

**Solution**:
```python
import pandas as pd
import psutil
from src.core.orchestrator import Orchestrator

def check_memory_usage():
    """Check current memory usage."""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    return memory.percent

def process_large_dataset(dataset_path, chunk_size=10000):
    """Process large dataset in chunks."""
    
    # Check memory before processing
    memory_usage = check_memory_usage()
    if memory_usage > 80:
        print("Warning: High memory usage detected")
    
    # Process in chunks
    chunks = []
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = process_chunk(chunk)
        chunks.append(processed_chunk)
        
        # Check memory after each chunk
        memory_usage = check_memory_usage()
        if memory_usage > 90:
            print("Memory usage too high, stopping processing")
            break
    
    # Combine processed chunks
    if chunks:
        combined_data = pd.concat(chunks, ignore_index=True)
        combined_data.to_csv("processed_large_dataset.csv", index=False)
        print("Large dataset processed successfully")
    
    return combined_data

def process_chunk(chunk):
    """Process a single chunk of data."""
    # Add your chunk processing logic here
    return chunk

# Usage
large_dataset = process_large_dataset("large_dataset.csv")
```

### Example 3: Model Performance Issues

**Problem**: Models are performing poorly.

**Solution**:
```python
from src.core.orchestrator import Orchestrator
from src.agents.data_agent import DataAgent
import pandas as pd
import numpy as np

def diagnose_model_performance(dataset_path):
    """Diagnose why models are performing poorly."""
    
    # Load and analyze data
    df = pd.read_csv(dataset_path)
    
    print("Data Analysis:")
    print(f"- Shape: {df.shape}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Duplicate rows: {df.duplicated().sum()}")
    
    # Check target distribution
    target_col = "target"  # Adjust as needed
    if target_col in df.columns:
        print(f"\nTarget Distribution:")
        print(df[target_col].value_counts())
        
        # Check for class imbalance
        class_counts = df[target_col].value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        if imbalance_ratio > 10:
            print(f"Warning: Severe class imbalance (ratio: {imbalance_ratio:.2f})")
    
    # Check feature correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlations = df[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr = correlations.iloc[i, j]
                if abs(corr) > 0.9:
                    high_corr_pairs.append((correlations.columns[i], correlations.columns[j], corr))
        
        if high_corr_pairs:
            print(f"\nHigh Correlation Pairs:")
            for col1, col2, corr in high_corr_pairs:
                print(f"- {col1} & {col2}: {corr:.4f}")
    
    # Feature importance analysis
    orchestrator = Orchestrator("src/config/config.json")
    result = orchestrator.run_pipeline(dataset_path)
    
    if result['success']:
        model = result['model_artifact']
        
        # Get feature importance
        from src.agents.model_agent import ModelAgent
        config = {"random_state": 42}
        logger = logging.getLogger("diagnosis")
        model_agent = ModelAgent(config, logger)
        
        importance = model_agent.get_feature_importance(
            model.model, 
            model.feature_names
        )
        
        print(f"\nFeature Importance:")
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"- {feature}: {score:.4f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if model.metrics.get('accuracy', 0) < 0.7:
            print("- Consider feature engineering")
            print("- Check for data quality issues")
            print("- Try different algorithms")
        
        if imbalance_ratio > 5:
            print("- Address class imbalance")
            print("- Use stratified sampling")
            print("- Consider ensemble methods")

# Usage
diagnose_model_performance("poor_performing_dataset.csv")
```

---

*These examples are regularly updated to reflect the latest features and best practices of AutoAI AgentHub.*
