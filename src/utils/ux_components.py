import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List, Optional
import json


class UXComponents:
    """Enhanced UX components for the AutoAI framework."""
    
    @staticmethod
    def create_progress_indicator(steps: List[str], current_step: int):
        """Create a visual progress indicator."""
        st.markdown("### 🚀 Pipeline Progress")
        
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if i < current_step:
                    st.markdown(f"✅ **{step}**")
                elif i == current_step:
                    st.markdown(f"🔄 **{step}**")
                else:
                    st.markdown(f"⏳ {step}")
    
    @staticmethod
    def create_metric_cards(metrics: Dict[str, float], title: str = "Performance Metrics"):
        """Create beautiful metric cards."""
        st.markdown(f"### 📊 {title}")
        
        # Determine number of columns based on metrics count
        n_metrics = len(metrics)
        n_cols = min(4, n_metrics)
        
        cols = st.columns(n_cols)
        
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % n_cols]:
                # Format metric name
                formatted_name = metric.replace('_', ' ').title()
                
                # Determine color based on metric type and value
                if 'accuracy' in metric.lower() or 'f1' in metric.lower() or 'r2' in metric.lower():
                    if isinstance(value, float) and value >= 0.8:
                        color = "🟢"
                    elif isinstance(value, float) and value >= 0.6:
                        color = "🟡"
                    else:
                        color = "🔴"
                elif 'error' in metric.lower() or 'mse' in metric.lower() or 'mae' in metric.lower():
                    if isinstance(value, float) and value <= 0.5:
                        color = "🟢"
                    elif isinstance(value, float) and value <= 1.0:
                        color = "🟡"
                    else:
                        color = "🔴"
                else:
                    color = "🔵"
                
                # Display metric
                if isinstance(value, float):
                    st.metric(
                        f"{color} {formatted_name}",
                        f"{value:.4f}",
                        help=f"Current value: {value:.6f}"
                    )
                else:
                    st.metric(f"{color} {formatted_name}", str(value))
    
    @staticmethod
    def create_data_quality_dashboard(df: pd.DataFrame):
        """Create a comprehensive data quality dashboard."""
        st.markdown("### 📈 Data Quality Dashboard")
        
        # Calculate quality metrics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_cols)) * 100
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100
        
        # Quality score calculation
        quality_score = 100 - min(missing_percentage, 30) - min(duplicate_percentage, 20)
        quality_score = max(quality_score, 0)
        
        # Create quality overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Rows", total_rows)
        with col2:
            st.metric("📋 Total Columns", total_cols)
        with col3:
            st.metric("❌ Missing Values", f"{missing_percentage:.1f}%")
        with col4:
            quality_color = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
            st.metric(f"{quality_color} Quality Score", f"{quality_score:.1f}/100")
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values heatmap
            if missing_values > 0:
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    fig = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title="Missing Values by Column",
                        labels={'x': 'Columns', 'y': 'Missing Count'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Data Types Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_model_comparison_chart(model_results: List[Dict[str, Any]]):
        """Create a comparison chart for multiple models."""
        if not model_results:
            return
        
        st.markdown("### 🏆 Model Comparison")
        
        # Extract model names and metrics
        model_names = [result['model_name'] for result in model_results]
        
        # Determine primary metric based on task type
        first_result = model_results[0]
        if 'accuracy' in first_result['metrics']:
            primary_metric = 'accuracy'
            metric_label = 'Accuracy'
        else:
            primary_metric = 'r2_score'
            metric_label = 'R² Score'
        
        primary_values = [result['metrics'].get(primary_metric, 0) for result in model_results]
        
        # Create comparison chart
        fig = px.bar(
            x=model_names,
            y=primary_values,
            title=f"Model Performance Comparison ({metric_label})",
            labels={'x': 'Models', 'y': metric_label},
            color=primary_values,
            color_continuous_scale='Viridis'
        )
        
        # Add value labels on bars
        fig.update_traces(texttemplate='%{y:.3f}', textposition='outside')
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create detailed metrics table
        st.markdown("#### 📋 Detailed Metrics")
        
        metrics_df = pd.DataFrame([
            {
                'Model': result['model_name'],
                **result['metrics']
            }
            for result in model_results
        ])
        
        # Format numeric columns
        numeric_columns = metrics_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            metrics_df[col] = metrics_df[col].round(4)
        
        st.dataframe(metrics_df, use_container_width=True)
    
    @staticmethod
    def create_feature_importance_chart(importance_dict: Dict[str, float], title: str = "Feature Importance"):
        """Create a feature importance visualization."""
        if not importance_dict:
            st.info("Feature importance not available for this model type.")
            return
        
        st.markdown(f"### 🎯 {title}")
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        # Create horizontal bar chart
        fig = px.bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            title="Feature Importance Scores",
            labels={'x': 'Importance', 'y': 'Features'}
        )
        
        fig.update_layout(height=max(400, len(features) * 30))
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_prediction_interface(model_artifact: Any, feature_names: List[str]):
        """Create an enhanced prediction interface."""
        st.markdown("### 🎯 Make Predictions")
        
        # Create input form
        input_data = {}
        
        # Group features by type for better organization
        numerical_features = []
        categorical_features = []
        
        for feature in feature_names:
            if any(cat in feature for cat in ['_', 'category', 'type']):
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)
        
        # Numerical features section
        if numerical_features:
            st.markdown("#### 📊 Numerical Features")
            cols = st.columns(min(3, len(numerical_features)))
            
            for i, feature in enumerate(numerical_features):
                with cols[i % len(cols)]:
                    # Determine default value and range based on feature name
                    if 'age' in feature.lower():
                        default_val = 35
                        min_val, max_val = 18, 100
                    elif 'income' in feature.lower() or 'salary' in feature.lower():
                        default_val = 50000
                        min_val, max_val = 0, 200000
                    elif 'score' in feature.lower() or 'rate' in feature.lower():
                        default_val = 0.5
                        min_val, max_val = 0.0, 1.0
                    else:
                        default_val = 0.0
                        min_val, max_val = -10.0, 10.0
                    
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=0.01,
                        key=f"num_{feature}"
                    )
        
        # Categorical features section
        if categorical_features:
            st.markdown("#### 🏷️ Categorical Features")
            cols = st.columns(min(3, len(categorical_features)))
            
            for i, feature in enumerate(categorical_features):
                with cols[i % len(cols)]:
                    # Handle one-hot encoded features
                    if '_' in feature:
                        base_name = feature.split('_')[0]
                        value = feature.split('_')[1]
                        input_data[feature] = st.selectbox(
                            f"{base_name.replace('_', ' ').title()}",
                            options=[0, 1],
                            format_func=lambda x: f"{value}" if x == 1 else "Other",
                            key=f"cat_{feature}"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            value=0.0,
                            step=0.01,
                            key=f"cat_{feature}"
                        )
        
        # Prediction button with enhanced styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Ensure all features are present
                    for feature in feature_names:
                        if feature not in input_df.columns:
                            input_df[feature] = 0.0
                    
                    # Reorder columns to match training data
                    input_df = input_df[feature_names]
                    
                    # Make prediction (this would need to be implemented with actual model)
                    # For now, we'll simulate a prediction
                    prediction = np.random.choice([0, 1]) if model_artifact.task_type == 'classification' else np.random.normal(0, 1)
                    
                    # Display result with enhanced styling
                    st.markdown("---")
                    
                    if model_artifact.task_type == 'classification':
                        prediction_text = f"Predicted Class: **{prediction}**"
                        confidence = np.random.uniform(0.7, 0.95)
                        st.success(f"🎯 {prediction_text}")
                        st.info(f"Confidence: {confidence:.2%}")
                    else:
                        st.success(f"🎯 Predicted Value: **{prediction:.4f}**")
                        st.info("This is a regression prediction.")
                    
                    # Show input summary
                    with st.expander("📋 Input Summary"):
                        st.json(input_data)
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    @staticmethod
    def create_sample_data_generator():
        """Create a sample data generator for testing."""
        st.markdown("### 🎲 Generate Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task_type = st.selectbox(
                "Task Type",
                ["Classification", "Regression"],
                help="Choose the type of machine learning task"
            )
        
        with col2:
            n_samples = st.slider(
                "Number of Samples",
                min_value=50,
                max_value=1000,
                value=200,
                step=50
            )
        
        n_features = st.slider(
            "Number of Features",
            min_value=2,
            max_value=10,
            value=5,
            step=1
        )
        
        if st.button("🎲 Generate Sample Dataset", type="primary"):
            # Generate sample data
            np.random.seed(42)
            
            if task_type == "Classification":
                X = np.random.normal(0, 1, (n_samples, n_features))
                y = np.random.choice([0, 1], n_samples)
            else:
                X = np.random.normal(0, 1, (n_samples, n_features))
                y = np.random.normal(0, 1, n_samples)
            
            # Create DataFrame
            feature_names = [f'feature_{i}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            # Display sample
            st.markdown("#### 📊 Generated Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Dataset",
                data=csv,
                file_name=f"sample_{task_type.lower()}_data.csv",
                mime="text/csv"
            )
    
    @staticmethod
    def create_help_section():
        """Create a comprehensive help section."""
        st.markdown("### ❓ Help & Documentation")
        
        with st.expander("📚 Getting Started"):
            st.markdown("""
            **Welcome to AutoAI AgentHub!** 🤖
            
            This tool automates the complete AI development pipeline:
            
            1. **📁 Upload Data**: Upload your CSV file
            2. **🎯 Select Target**: Choose or auto-detect the target column
            3. **🚀 Run Pipeline**: Click "Generate AI Model" to start
            4. **📊 View Results**: Review model performance and metrics
            5. **🎮 Test Model**: Use the generated interface to make predictions
            
            **Supported File Formats**: CSV files up to 10MB
            **Supported Tasks**: Classification and Regression
            **Supported Models**: Logistic Regression, Decision Trees, Random Forest, Linear Regression, Ridge Regression
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            
            - **File Upload Error**: Ensure your file is a valid CSV format
            - **No Target Detected**: Manually select the target column
            - **Poor Model Performance**: Try different datasets or check data quality
            - **Memory Issues**: Reduce dataset size or number of features
            
            **Performance Tips:**
            
            - Use datasets with 100-1000 rows for best results
            - Ensure target column has clear patterns
            - Remove unnecessary columns before upload
            - Check for missing values and outliers
            """)
        
        with st.expander("📊 Understanding Results"):
            st.markdown("""
            **Classification Metrics:**
            - **Accuracy**: Overall correctness (0-1, higher is better)
            - **F1-Score**: Harmonic mean of precision and recall (0-1, higher is better)
            - **Precision**: True positive rate (0-1, higher is better)
            - **Recall**: Sensitivity (0-1, higher is better)
            
            **Regression Metrics:**
            - **RMSE**: Root Mean Square Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            - **R² Score**: Coefficient of determination (0-1, higher is better)
            
            **Quality Score**: Overall data quality (0-100, higher is better)
            """)
        
        with st.expander("🎯 Best Practices"):
            st.markdown("""
            **Data Preparation:**
            - Clean your data before upload
            - Remove duplicate rows
            - Handle missing values appropriately
            - Ensure target column has sufficient variation
            
            **Model Selection:**
            - Try multiple algorithms
            - Compare performance metrics
            - Consider interpretability vs accuracy trade-offs
            - Validate on unseen data
            
            **Deployment:**
            - Test predictions with various inputs
            - Monitor model performance over time
            - Retrain with new data periodically
            - Document model assumptions and limitations
            """)


def apply_enhanced_styling():
    """Apply enhanced styling to the Streamlit app."""
    st.markdown("""
    <style>
    /* Enhanced styling for better UX */
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Progress indicator styling */
    .progress-step {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
    
    .progress-step.completed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .progress-step.current {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .progress-step.pending {
        background: #f0f0f0;
        color: #666;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    </style>
    """, unsafe_allow_html=True)
