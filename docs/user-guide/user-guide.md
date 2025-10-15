# AutoAI AgentHub - User Guide

## Welcome to AutoAI AgentHub! 🤖

AutoAI AgentHub is an intelligent framework that automates the complete AI development pipeline using specialized agents that collaborate to build, train, and deploy machine learning models with minimal human intervention.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Web Interface Guide](#web-interface-guide)
5. [Command Line Interface](#command-line-interface)
6. [Data Requirements](#data-requirements)
7. [Understanding Results](#understanding-results)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Getting Started

### What is AutoAI AgentHub?

AutoAI AgentHub is a multi-agent system that automates machine learning workflows:

- **📊 Data Agent**: Handles data preprocessing, cleaning, and preparation
- **🎯 Model Agent**: Trains multiple models and selects the best performer
- **🚀 Deployment Agent**: Creates interactive web applications for your models

### Key Features

- ✅ **One-Click Automation**: Upload data, get a trained model
- ✅ **Smart Preprocessing**: Automatic data cleaning and feature engineering
- ✅ **Multiple Algorithms**: Tests various ML algorithms automatically
- ✅ **Interactive Deployment**: Generates Streamlit apps for model testing
- ✅ **Comprehensive Metrics**: Detailed performance analysis
- ✅ **User-Friendly**: No coding required for basic usage

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB+ RAM recommended
- 2GB+ free disk space

### Step-by-Step Installation

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd AutoAI-AgentHub
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python main.py --help
   ```

4. **Run Automated Setup (Recommended)**
   ```bash
   python scripts/deploy.py
   ```

### Installation Verification

After installation, test with sample data:

```bash
python main.py --demo
```

This will run a demonstration with sample data to verify everything works correctly.

## Quick Start

### Option 1: Web Interface (Recommended for Beginners)

1. **Launch the Web Interface**
   ```bash
   python main.py --web
   ```

2. **Open Your Browser**
   - Navigate to `http://localhost:8501`
   - You'll see the AutoAI AgentHub interface

3. **Upload Your Data**
   - Click "Browse files" or drag and drop your CSV file
   - Supported formats: CSV files up to 10MB

4. **Configure Settings** (Optional)
   - Target Column: Select or let the system auto-detect
   - Task Type: Choose Classification or Regression
   - Test Size: Adjust train/test split (default: 20%)

5. **Generate Your Model**
   - Click "🤖 Generate AI Model"
   - Wait for the pipeline to complete (usually 1-3 minutes)

6. **Explore Results**
   - View model performance metrics
   - Test predictions with the generated interface
   - Download model artifacts

### Option 2: Command Line Interface

1. **Basic Usage**
   ```bash
   python main.py --dataset your_data.csv --target target_column
   ```

2. **With Task Type Specification**
   ```bash
   python main.py --dataset iris.csv --target species --task classification
   ```

3. **Auto-detect Target**
   ```bash
   python main.py --dataset iris.csv
   ```

## Web Interface Guide

### Main Dashboard

The web interface provides an intuitive drag-and-drop experience:

#### Header Section
- **🤖 AutoAI AgentHub**: Main title and branding
- **Status Indicator**: Shows current system status
- **Help Button**: Access to documentation and examples

#### File Upload Area
- **Drag & Drop Zone**: Drop CSV files here
- **Browse Button**: Click to select files from your computer
- **File Validation**: Automatic file format and size checking

#### Configuration Panel
- **Target Column**: Dropdown to select target variable
- **Auto-Detect**: Button to automatically detect the target
- **Task Type**: Choose between Classification and Regression
- **Advanced Settings**: Expandable section for fine-tuning

#### Results Section
- **Model Performance**: Visual metrics and charts
- **Feature Importance**: Bar charts showing feature relevance
- **Prediction Interface**: Interactive form for testing predictions
- **Download Options**: Export model artifacts and reports

### Sidebar Navigation

#### Quick Actions
- **📁 Upload New Dataset**: Start a new analysis
- **📊 View Results**: Access previous results
- **⚙️ Settings**: Configure system preferences
- **❓ Help**: Access documentation and tutorials

#### Sample Datasets
- **Iris Dataset**: Classic classification example
- **California Housing**: Regression example
- **Wine Dataset**: Multi-class classification
- **Custom Examples**: Additional sample data

### Understanding the Interface

#### Progress Indicators
- **Pipeline Status**: Shows current processing stage
- **Progress Bar**: Visual indication of completion
- **Time Estimates**: Expected completion time

#### Error Messages
- **File Format Errors**: Clear guidance on supported formats
- **Data Quality Issues**: Specific recommendations for data cleaning
- **Processing Errors**: Detailed error descriptions and solutions

## Command Line Interface

### Basic Commands

#### Run with Dataset
```bash
python main.py --dataset path/to/data.csv
```

#### Specify Target Column
```bash
python main.py --dataset data.csv --target target_column
```

#### Specify Task Type
```bash
python main.py --dataset data.csv --target target --task classification
```

#### Launch Web Interface
```bash
python main.py --web
```

#### Run Demo
```bash
python main.py --demo
```

### Advanced Options

#### Custom Configuration
```bash
python main.py --dataset data.csv --config custom_config.json
```

#### Verbose Output
```bash
python main.py --dataset data.csv --verbose
```

#### Help and Information
```bash
python main.py --help
python main.py --version
```

### Command Line Output

The CLI provides detailed feedback:

```
🤖 AutoAI AgentHub - Starting Pipeline
==================================================
📁 Dataset: data/iris.csv
🎯 Target: species
🔧 Task: classification
📊 Test Size: 0.2

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

🚀 Generating Deployment Interface...
✅ Streamlit app created: artifacts/streamlit/streamlit_app_20251015.py
✅ API created: artifacts/api/api_20251015.py

📊 Results Summary:
- Model: Random Forest
- Accuracy: 97.0%
- F1-Score: 0.97
- Processing Time: 45 seconds

🎉 Pipeline completed successfully!
```

## Data Requirements

### Supported File Formats

- **CSV Files**: Comma-separated values (recommended)
- **Excel Files**: .xlsx format (basic support)
- **Text Files**: Tab-separated values (.tsv)

### Data Quality Requirements

#### Minimum Requirements
- **Rows**: At least 10 rows (recommended: 100+)
- **Columns**: At least 2 columns (1 feature + 1 target)
- **File Size**: Maximum 10MB

#### Data Types Supported
- **Numerical**: Integers, floats, decimals
- **Categorical**: Strings, text, categories
- **Boolean**: True/False values
- **Dates**: Basic date formats (limited support)

#### Data Quality Guidelines

##### Good Data Characteristics
- ✅ **Clean Data**: No missing values or minimal missing data
- ✅ **Consistent Format**: Uniform data types within columns
- ✅ **Meaningful Names**: Descriptive column names
- ✅ **Balanced Classes**: For classification, relatively balanced target classes
- ✅ **Relevant Features**: Features that relate to the target variable

##### Common Data Issues
- ❌ **Too Many Missing Values**: >50% missing in important columns
- ❌ **Inconsistent Formats**: Mixed data types in single columns
- ❌ **Irrelevant Features**: Columns unrelated to the target
- ❌ **Duplicate Rows**: Excessive duplicate entries
- ❌ **Outliers**: Extreme values that skew results

### Data Preparation Tips

#### Before Uploading
1. **Remove Unnecessary Columns**: ID columns, timestamps, etc.
2. **Handle Missing Values**: Fill or remove missing data
3. **Standardize Formats**: Ensure consistent data types
4. **Check for Duplicates**: Remove duplicate rows
5. **Validate Target Column**: Ensure target is clearly defined

#### Example Data Structure

**Good Example (Iris Dataset):**
```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
```

**Poor Example:**
```
id,timestamp,feature1,feature2,target,notes
1,2023-01-01,5.1,3.5,setosa,good sample
2,2023-01-02,,3.0,,bad sample
3,2023-01-03,4.7,3.2,virginica,
```

## Understanding Results

### Model Performance Metrics

#### Classification Metrics
- **Accuracy**: Overall correctness (0-1, higher is better)
- **F1-Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **Precision**: True positive rate (0-1, higher is better)
- **Recall**: Sensitivity (0-1, higher is better)

#### Regression Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R² Score**: Coefficient of determination (0-1, higher is better)

### Interpreting Results

#### Good Performance Indicators
- **Classification**: Accuracy > 0.8, F1-Score > 0.8
- **Regression**: R² > 0.7, RMSE < 20% of target range

#### Performance Levels
- **Excellent**: > 90% accuracy (classification) or R² > 0.9
- **Good**: 80-90% accuracy or R² 0.7-0.9
- **Fair**: 70-80% accuracy or R² 0.5-0.7
- **Poor**: < 70% accuracy or R² < 0.5

### Feature Importance

The system provides feature importance scores showing which features contribute most to predictions:

- **High Importance**: Features that strongly influence predictions
- **Medium Importance**: Features with moderate influence
- **Low Importance**: Features with minimal influence

### Model Comparison

When multiple models are trained, the system shows:
- **Performance Ranking**: Best to worst performing models
- **Trade-offs**: Speed vs. accuracy comparisons
- **Recommendations**: Suggested model for your use case

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```json
{
  "max_file_size_mb": 20,
  "default_test_size": 0.3,
  "random_state": 123,
  "max_categorical_cardinality": 100,
  "streamlit_port": 8502,
  "model_trials": {
    "classification": [
      "LogisticRegression",
      "RandomForestClassifier",
      "GradientBoostingClassifier"
    ],
    "regression": [
      "LinearRegression",
      "RandomForestRegressor",
      "GradientBoostingRegressor"
    ]
  }
}
```

### Batch Processing

Process multiple datasets:

```python
import os
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator("src/config/config.json")

datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
results = []

for dataset in datasets:
    if os.path.exists(f"data/{dataset}"):
        result = orchestrator.run_pipeline(f"data/{dataset}")
        results.append({
            "dataset": dataset,
            "success": result['success'],
            "accuracy": result['model_artifact'].metrics.get('accuracy', 0)
        })
```

### Custom Model Integration

Add custom models to the framework:

1. **Update Configuration**
   ```json
   {
     "model_trials": {
       "classification": [
         "LogisticRegression",
         "DecisionTreeClassifier",
         "RandomForestClassifier",
         "CustomModel"
       ]
     }
   }
   ```

2. **Implement Custom Model**
   ```python
   from sklearn.base import BaseEstimator, ClassifierMixin
   
   class CustomModel(BaseEstimator, ClassifierMixin):
       def fit(self, X, y):
           # Custom training logic
           return self
       
       def predict(self, X):
           # Custom prediction logic
           return predictions
   ```

### API Integration

Use the generated API endpoints:

```python
import requests

# Make predictions via API
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [5.1, 3.5, 1.4, 0.2],
        "model_type": "RandomForestClassifier"
    }
)

predictions = response.json()
print(f"Prediction: {predictions['prediction']}")
```

## Troubleshooting

### Common Issues and Solutions

#### File Upload Problems

**Issue**: "File format not supported"
- **Solution**: Ensure file is CSV format with proper encoding
- **Check**: File extension, encoding (UTF-8 recommended)

**Issue**: "File too large"
- **Solution**: Reduce file size or increase limit in configuration
- **Check**: Current limit is 10MB by default

#### Data Processing Errors

**Issue**: "No target column detected"
- **Solution**: Manually specify target column or ensure clear target variable
- **Check**: Target column has distinct values

**Issue**: "Too many missing values"
- **Solution**: Clean data before upload or adjust missing value threshold
- **Check**: Missing value percentage in important columns

#### Model Training Issues

**Issue**: "All models failed to train"
- **Solution**: Check data quality and feature-target relationship
- **Check**: Sufficient data, proper data types, meaningful features

**Issue**: "Poor model performance"
- **Solution**: Improve data quality, add more features, or adjust task type
- **Check**: Feature relevance, data balance, target definition

#### Deployment Problems

**Issue**: "Streamlit app not launching"
- **Solution**: Check port availability and Streamlit installation
- **Check**: Port 8501 is free, Streamlit is installed

**Issue**: "API not responding"
- **Solution**: Verify FastAPI installation and port configuration
- **Check**: Port 8000 is free, FastAPI is installed

### Error Messages Guide

#### File-Related Errors
- `FileNotFoundError`: File path is incorrect
- `PermissionError`: Insufficient file permissions
- `UnicodeDecodeError`: File encoding issues

#### Data-Related Errors
- `ValueError`: Invalid data format or values
- `KeyError`: Missing required columns
- `TypeError`: Incorrect data types

#### Processing Errors
- `MemoryError`: Insufficient memory for processing
- `TimeoutError`: Processing took too long
- `RuntimeError`: Unexpected processing failure

### Getting Help

#### Self-Help Resources
1. **Check Documentation**: Review this user guide
2. **Run Diagnostics**: Use `python main.py --demo` to test system
3. **Check Logs**: Review log files in `artifacts/logs/`
4. **Validate Data**: Ensure data meets requirements

#### Support Channels
1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Comprehensive guides and examples
3. **Community**: User forums and discussions
4. **Email Support**: Direct technical support

## Best Practices

### Data Preparation

#### Before Upload
1. **Clean Your Data**: Remove duplicates, handle missing values
2. **Standardize Formats**: Ensure consistent data types
3. **Validate Target**: Ensure target variable is clearly defined
4. **Feature Engineering**: Create meaningful features
5. **Data Splitting**: Consider temporal or spatial splits

#### Data Quality Checks
1. **Missing Values**: Check percentage and patterns
2. **Outliers**: Identify and handle extreme values
3. **Data Types**: Ensure appropriate data types
4. **Consistency**: Check for inconsistent formats
5. **Balance**: Ensure balanced classes for classification

### Model Selection

#### Choose Appropriate Algorithms
- **Linear Models**: Good for interpretable results
- **Tree Models**: Good for non-linear relationships
- **Ensemble Models**: Good for high accuracy
- **Custom Models**: Good for domain-specific problems

#### Performance Optimization
1. **Feature Selection**: Remove irrelevant features
2. **Hyperparameter Tuning**: Optimize model parameters
3. **Cross-Validation**: Use proper validation strategies
4. **Ensemble Methods**: Combine multiple models
5. **Regularization**: Prevent overfitting

### Deployment Considerations

#### Model Monitoring
1. **Performance Tracking**: Monitor model performance over time
2. **Data Drift**: Detect changes in input data distribution
3. **Model Updates**: Plan for model retraining
4. **A/B Testing**: Compare different model versions
5. **Feedback Loop**: Collect user feedback for improvements

#### Production Readiness
1. **Scalability**: Ensure system can handle production load
2. **Reliability**: Implement error handling and fallbacks
3. **Security**: Protect sensitive data and models
4. **Documentation**: Maintain comprehensive documentation
5. **Testing**: Implement comprehensive testing strategies

### Performance Tips

#### System Optimization
1. **Memory Management**: Monitor and optimize memory usage
2. **Processing Speed**: Use parallel processing where possible
3. **Storage**: Optimize file storage and access patterns
4. **Caching**: Implement caching for repeated operations
5. **Resource Monitoring**: Track system resource usage

#### Best Practices Summary
- ✅ **Start Simple**: Begin with basic models and iterate
- ✅ **Validate Results**: Always validate model performance
- ✅ **Document Everything**: Keep records of experiments and results
- ✅ **Monitor Performance**: Track model performance over time
- ✅ **Stay Updated**: Keep up with latest ML techniques and tools

---

*This user guide is regularly updated to reflect the latest features and best practices of AutoAI AgentHub.*
