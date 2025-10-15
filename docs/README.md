# AutoAI AgentHub - Documentation

Welcome to the comprehensive documentation for AutoAI AgentHub, an intelligent framework that automates the complete AI development pipeline using specialized agents.

## 📚 Documentation Overview

AutoAI AgentHub provides a multi-agent system that collaborates to process data, train models, and deploy applications with minimal human intervention. This documentation covers all aspects of the framework from basic usage to advanced customization.

## 🚀 Quick Navigation

### Getting Started
- **[User Guide](user-guide/user-guide.md)** - Complete user manual with step-by-step instructions
- **[Examples & Tutorials](examples/examples.md)** - Practical examples and code samples
- **[API Documentation](api/api-documentation.md)** - Complete API reference

### Technical Documentation
- **[Architecture](architecture/architecture.md)** - System design and technical architecture
- **[Configuration Guide](#configuration-guide)** - System configuration and customization
- **[Deployment Guide](#deployment-guide)** - Production deployment instructions

## 📖 Documentation Sections

### 1. [User Guide](user-guide/user-guide.md)
Comprehensive user manual covering:
- **Installation & Setup**: Step-by-step installation instructions
- **Web Interface**: Complete guide to the Streamlit interface
- **Command Line Interface**: CLI usage and options
- **Data Requirements**: Data format and quality guidelines
- **Understanding Results**: Interpreting model performance
- **Troubleshooting**: Common issues and solutions
- **Best Practices**: Tips for optimal usage

### 2. [API Documentation](api/api-documentation.md)
Complete API reference including:
- **Core Components**: Orchestrator, Agents, Data Structures
- **Agent Classes**: DataAgent, ModelAgent, DeployAgent
- **Utility Functions**: Validation, file operations, configuration
- **Configuration**: System configuration parameters
- **Error Handling**: Exception handling and recovery
- **Examples**: Code examples and usage patterns

### 3. [Architecture Documentation](architecture/architecture.md)
Technical architecture overview:
- **System Overview**: High-level architecture and design principles
- **System Components**: Core components and their responsibilities
- **Agent Architecture**: Individual agent design and lifecycle
- **Data Flow**: Pipeline data flow and transformations
- **Communication Patterns**: Inter-agent communication
- **Deployment Architecture**: Application deployment strategies
- **Scalability**: Horizontal and vertical scaling considerations
- **Security**: Security architecture and best practices
- **Performance**: Performance optimization and monitoring

### 4. [Examples & Tutorials](examples/examples.md)
Practical examples and tutorials:
- **Quick Start Examples**: Basic classification and regression
- **Classification Examples**: Customer churn, spam detection, multi-class
- **Regression Examples**: Sales forecasting, stock prediction, energy consumption
- **Advanced Examples**: Custom models, batch processing, real-time services
- **Custom Configuration**: Performance, research, and production configs
- **API Usage**: RESTful integration and client examples
- **Troubleshooting**: Common problems and solutions

## 🛠️ Configuration Guide

### Basic Configuration
The system uses `src/config/config.json` for configuration:

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

### Custom Configuration Examples

#### High-Performance Configuration
```json
{
  "max_file_size_mb": 100,
  "model_trials": {
    "classification": [
      "LogisticRegression",
      "DecisionTreeClassifier", 
      "RandomForestClassifier",
      "GradientBoostingClassifier",
      "SVC"
    ]
  },
  "performance": {
    "n_jobs": -1,
    "cross_validation_folds": 5
  }
}
```

#### Research Configuration
```json
{
  "default_test_size": 0.3,
  "log_level": "DEBUG",
  "model_trials": {
    "classification": [
      "LogisticRegression",
      "DecisionTreeClassifier",
      "RandomForestClassifier",
      "GradientBoostingClassifier",
      "AdaBoostClassifier",
      "ExtraTreesClassifier"
    ]
  },
  "research": {
    "save_intermediate_results": true,
    "detailed_logging": true,
    "cross_validation_folds": 10
  }
}
```

## 🚀 Deployment Guide

### Local Deployment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Automated Setup**
   ```bash
   python scripts/deploy.py
   ```

3. **Launch Application**
   ```bash
   # Web Interface
   python main.py --web
   
   # Command Line
   python main.py --dataset your_data.csv --target target_column
   ```

### Production Deployment

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   ```bash
   # Copy and customize configuration
   cp src/config/config.json production_config.json
   # Edit production_config.json for production settings
   ```

3. **Service Deployment**
   ```bash
   # Using systemd (Linux)
   sudo systemctl enable autoai-agenthub
   sudo systemctl start autoai-agenthub
   
   # Using Docker
   docker build -t autoai-agenthub .
   docker run -p 8501:8501 autoai-agenthub
   ```

### Cloud Deployment

#### AWS Deployment
```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --user-data file://user-data.sh
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["python", "main.py", "--web"]
```

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **Python**: 3.10 or higher
- **RAM**: 16GB or higher
- **Storage**: 10GB free space (SSD recommended)
- **CPU**: Multi-core processor (4+ cores)

### Dependencies
- **Core**: pandas, numpy, scikit-learn, joblib
- **Web**: streamlit, fastapi, uvicorn
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest, pytest-cov
- **Development**: black, flake8

## 🔧 Development Guide

### Setting Up Development Environment

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd AutoAI-AgentHub
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   pytest tests/ --cov=src  # With coverage
   ```

### Code Structure
```
AutoAI-AgentHub/
├── src/                    # Source code
│   ├── agents/            # Agent implementations
│   ├── core/              # Core system components
│   ├── utils/             # Utility functions
│   └── config/            # Configuration files
├── tests/                  # Test suite
├── docs/                   # Documentation
├── data/                   # Sample datasets
├── scripts/                # Deployment scripts
└── artifacts/              # Generated artifacts
```

### Contributing Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Write tests for new features
3. **Documentation**: Update documentation for changes
4. **Commits**: Use descriptive commit messages
5. **Pull Requests**: Provide clear descriptions

## 📞 Support & Community

### Getting Help

1. **Documentation**: Check this documentation first
2. **GitHub Issues**: Report bugs and request features
3. **Community Forum**: Join discussions and ask questions
4. **Email Support**: Direct technical support

### Reporting Issues

When reporting issues, please include:
- **System Information**: OS, Python version, package versions
- **Error Messages**: Complete error messages and stack traces
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened

### Feature Requests

For feature requests, please include:
- **Use Case**: Describe the problem you're trying to solve
- **Proposed Solution**: How you think it should work
- **Alternatives**: Other solutions you've considered
- **Additional Context**: Any other relevant information

## 📄 License & Legal

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation library
- **NumPy**: Numerical computing library

### Disclaimer
This software is provided "as is" without warranty of any kind. Use at your own risk.

---

## 🔄 Documentation Updates

This documentation is regularly updated to reflect the latest features and changes. Check the version history for recent updates.

**Last Updated**: October 15, 2025  
**Version**: 1.0  
**Status**: Production Ready

---

*For the most up-to-date information, please refer to the latest version of this documentation.*
