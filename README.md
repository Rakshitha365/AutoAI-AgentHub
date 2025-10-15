# 🤖 AutoAI AgentHub

**Automated AI Model Development with Multi-Agent Collaboration**

An intelligent framework that automates the complete AI development pipeline using specialized agents that collaborate to build, train, and deploy machine learning models with minimal human intervention.

## 🌟 Features

- **🤖 Multi-Agent Architecture**: Specialized agents handle different aspects of AI development
- **📊 Automated Data Processing**: Intelligent data cleaning, preprocessing, and feature engineering
- **🎯 Smart Model Selection**: Automatic model training and selection for classification and regression
- **🚀 One-Click Deployment**: Instant Streamlit interface generation for model demonstration
- **📈 Comprehensive Metrics**: Detailed performance analysis and reporting
- **🎨 User-Friendly Interface**: Intuitive web interface for non-technical users

## 🏗️ Architecture

The framework consists of three specialized agents:

### 1. 📊 Data Agent
- **CSV Upload & Validation**: Handles file uploads with size and format validation
- **Missing Value Imputation**: Intelligent handling of missing data
- **Categorical Encoding**: Automatic encoding of categorical variables
- **Feature Scaling**: Standardization and normalization of numerical features
- **Train-Test Splitting**: Proper data splitting with stratification

### 2. 🎯 Model Agent
- **Task Detection**: Automatic classification vs regression detection
- **Model Training**: Multiple algorithm training and evaluation
- **Performance Metrics**: Comprehensive evaluation metrics
- **Best Model Selection**: Intelligent model selection based on performance
- **Model Persistence**: Save and load trained models

### 3. 🚀 Deployment Agent
- **Streamlit Interface**: Dynamic UI generation based on model metadata
- **Prediction Interface**: Real-time prediction capabilities
- **Model Reporting**: Comprehensive model documentation
- **API Generation**: Optional API endpoint creation

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AutoAI-AgentHub
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run automated setup (Recommended)**
   ```bash
   python scripts/deploy.py
   ```

4. **Launch the application**
   
   **Option A: Streamlit Interface (Recommended)**
   ```bash
   python main.py --web
   ```
   
   **Option B: Command Line Interface**
   ```bash
   python main.py --dataset your_data.csv --target target_column
   ```
   
   **Option C: Demo with sample data**
   ```bash
   python main.py --demo
   ```

### Usage

1. **Upload Dataset**: Upload a CSV file through the web interface
2. **Select Target**: Choose or auto-detect the target column
3. **Run Pipeline**: Click "Generate AI Model" to start the automation
4. **View Results**: Review model performance and metrics
5. **Test Model**: Use the generated Streamlit app to make predictions

## 📁 Project Structure

```
AutoAI-AgentHub/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py          # Package initialization
│   │   ├── data_agent.py        # Data processing agent
│   │   ├── model_agent.py       # Model training agent
│   │   └── deploy_agent.py      # Deployment agent
│   ├── core/                     # Core system components
│   │   ├── __init__.py          # Package initialization
│   │   ├── orchestrator.py      # Main orchestrator
│   │   └── dataclasses.py       # Data structures
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py          # Package initialization
│   │   ├── validation.py        # Validation utilities
│   │   └── ux_components.py     # UX components
│   ├── config/                   # Configuration files
│   │   └── config.json          # Main configuration
│   ├── __init__.py              # Main package initialization
│   └── streamlit_app.py         # Web interface
├── tests/                        # Test suite
│   ├── test_data_agent.py       # Data agent unit tests
│   ├── test_model_agent.py      # Model agent unit tests
│   ├── test_integration.py      # Integration tests
│   └── benchmark_performance.py # Performance benchmarks
├── data/                         # Sample datasets
│   ├── generate_datasets.py     # Dataset generator
│   ├── iris.csv                  # Iris classification dataset
│   ├── california_housing.csv   # California housing regression dataset
│   ├── wine.csv                  # Wine classification dataset
│   ├── breast_cancer.csv         # Breast cancer dataset
│   └── [5 more sample datasets]  # Additional sample data
├── scripts/                      # Deployment scripts
│   └── deploy.py                # Automated deployment
├── artifacts/                    # Generated artifacts
│   ├── models/                   # Trained models
│   ├── reports/                  # Performance reports
│   ├── logs/                     # System logs
│   └── streamlit/               # Generated apps
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The framework can be configured through `src/config/config.json`:

```json
{
  "max_file_size_mb": 10,
  "default_test_size": 0.2,
  "random_state": 42,
  "max_categorical_cardinality": 50,
  "streamlit_port": 8501,
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

## 🧪 Testing

The framework includes a comprehensive test suite with **100% passing tests**:

```bash
# Run all tests (30 tests)
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_data_agent.py

# Run integration tests
pytest tests/test_integration.py

# Run performance benchmarks
pytest tests/benchmark_performance.py
```

### Test Coverage
- ✅ **Unit Tests**: Individual agent testing
- ✅ **Integration Tests**: End-to-end pipeline testing  
- ✅ **Performance Tests**: Benchmarking and optimization
- ✅ **Error Handling**: Comprehensive error scenario testing

## 📊 Supported Algorithms

### Classification
- **Logistic Regression**: Linear classification with regularization
- **Decision Tree**: Non-linear classification with interpretability
- **Random Forest**: Ensemble method with high accuracy

### Regression
- **Linear Regression**: Basic linear regression
- **Ridge Regression**: Linear regression with L2 regularization
- **Random Forest Regressor**: Ensemble regression method

## 📈 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination

## 🎯 Use Cases

### Academic Projects
- **Student Projects**: Quick AI model development for coursework
- **Research Prototypes**: Rapid prototyping for research ideas
- **Educational Tool**: Learning AI concepts through automation
- **Course Submissions**: Complete, professional-grade projects

### Business Applications
- **Proof of Concepts**: Quick validation of AI ideas
- **Data Exploration**: Understanding data patterns and relationships
- **Model Comparison**: Comparing different algorithms on the same dataset
- **Rapid Prototyping**: Fast iteration and validation

### Production Ready
- **Deployment Ready**: Complete system with all components
- **Scalable Architecture**: Modular design for easy extension
- **Professional Quality**: Production-grade code and documentation

## 🔍 Example Workflow

1. **Data Upload**: Upload `iris.csv` dataset
2. **Auto-Detection**: System detects "species" as classification target
3. **Processing**: Data is cleaned, encoded, and split
4. **Training**: Three classification models are trained and evaluated
5. **Selection**: Best model is selected based on F1-score
6. **Deployment**: Streamlit app is generated with prediction interface
7. **Testing**: Users can input flower measurements and get species predictions

## 🛠️ Development

### Adding New Models

To add new models to the framework:

1. **Update Configuration**: Add model to `config.json`
2. **Implement Model**: Add model to `ModelAgent` class
3. **Add Tests**: Create test cases for the new model
4. **Update Documentation**: Document the new model capabilities

### Extending Agents

Each agent can be extended with new functionality:

- **Data Agent**: Add new preprocessing techniques
- **Model Agent**: Add new algorithms or evaluation metrics
- **Deployment Agent**: Add new deployment options

## 📚 Documentation

- **📖 [Complete Documentation](docs/README.md)**: Comprehensive documentation hub
- **👤 [User Guide](docs/user-guide/user-guide.md)**: Step-by-step user manual
- **🔧 [API Documentation](docs/api/api-documentation.md)**: Complete API reference
- **🏗️ [Architecture](docs/architecture/architecture.md)**: Technical architecture overview
- **💡 [Examples & Tutorials](docs/examples/examples.md)**: Practical examples and code samples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request