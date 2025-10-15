import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from src.utils.validation import create_sample_data


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="AutoAI AgentHub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AutoAI AgentHub</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Automated Machine Learning Pipeline with Multi-Agent Collaboration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize orchestrator
    try:
        config_path = "src/config/config.json"
        orchestrator = Orchestrator(config_path)
    except Exception as e:
        st.error(f"❌ Error initializing framework: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # File upload section
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with your dataset. Maximum size: 10MB"
    )
    
    dataset_df = None
    if uploaded_file is not None:
        try:
            dataset_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Dataset uploaded successfully!")
            st.sidebar.info(f"📊 Shape: {dataset_df.shape[0]} rows, {dataset_df.shape[1]} columns")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading CSV: {e}")
    
    # Configuration section
    st.sidebar.header("⚙️ Configuration")
    
    if dataset_df is not None:
        # Target column selection
        all_columns = dataset_df.columns.tolist()
        target_column = st.sidebar.selectbox(
            "🎯 Select Target Column",
            all_columns,
            help="Choose the column you want to predict"
        )
        
        # Task type selection
        task_hint = st.sidebar.selectbox(
            "📊 Task Type (Optional)",
            ["Auto-detect", "classification", "regression"],
            help="Let the system auto-detect or specify manually"
        )
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 1000, 42)
    else:
        target_column = None
        task_hint = "Auto-detect"
        test_size = 0.2
        random_state = 42
    
    # Main content area
    if dataset_df is not None:
        # Dataset preview
        st.header("📊 Dataset Preview")
        st.dataframe(dataset_df.head(10), use_container_width=True)
        
        # Data quality analysis
        st.header("🔍 Data Quality Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Rows", len(dataset_df))
        with col2:
            st.metric("📋 Total Columns", len(dataset_df.columns))
        with col3:
            missing_count = dataset_df.isnull().sum().sum()
            st.metric("❌ Missing Values", missing_count)
        with col4:
            quality_score = max(0, 100 - (missing_count / (len(dataset_df) * len(dataset_df.columns)) * 100))
            quality_color = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
            st.metric(f"{quality_color} Quality Score", f"{quality_score:.1f}/100")
        
        # Run pipeline button
        st.sidebar.header("🚀 Run Pipeline")
        
        if st.sidebar.button("🤖 Generate AI Model", type="primary", use_container_width=True):
            run_pipeline(orchestrator, uploaded_file, target_column, task_hint, test_size, random_state)
    
    else:
        # Welcome screen
        st.header("👋 Welcome to AutoAI AgentHub")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What is AutoAI?
            
            AutoAI is an intelligent framework that automates the complete machine learning pipeline:
            
            - **📊 Data Processing**: Automatic data cleaning and preprocessing
            - **🤖 Model Training**: Multiple algorithms trained and evaluated
            - **🚀 Deployment**: Instant web interface generation
            - **📈 Analysis**: Comprehensive performance metrics
            
            ### 🚀 How to Get Started
            
            1. **Upload your dataset** using the sidebar
            2. **Select target column** you want to predict
            3. **Click "Generate AI Model"** to start automation
            4. **Review results** and test predictions
            """)
        
        with col2:
            st.markdown("""
            ### 📋 Supported Features
            
            **Data Types:**
            - CSV files up to 10MB
            - Classification and regression tasks
            - Automatic data type detection
            
            **Algorithms:**
            - Logistic Regression
            - Decision Trees
            - Random Forest
            - Linear Regression
            - Ridge Regression
            
            **Outputs:**
            - Trained ML models
            - Interactive web interface
            - Performance metrics
            - Prediction API
            """)
        
        # Sample data generator
        st.header("🎲 Try with Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Classification Sample", use_container_width=True):
                sample_data = create_sample_data(200, 5, 'classification')
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample Dataset",
                    data=csv,
                    file_name="sample_classification.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Generate Regression Sample", use_container_width=True):
                sample_data = create_sample_data(200, 5, 'regression')
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample Dataset",
                    data=csv,
                    file_name="sample_regression.csv",
                    mime="text/csv"
                )


def run_pipeline(orchestrator, uploaded_file, target_column, task_hint, test_size, random_state):
    """Run the complete AI pipeline."""
    st.header("🤖 AI Pipeline Execution")
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        temp_file_path = None
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file_path = f.name
            uploaded_file.seek(0)
            f.write(uploaded_file.getvalue().decode('utf-8'))
        
        status_text.info("🔄 Starting pipeline...")
        progress_bar.progress(10)
        
        # Update configuration
        config = orchestrator.config.copy()
        config["default_test_size"] = test_size
        config["random_state"] = random_state
        
        # Run pipeline
        result = orchestrator.run_pipeline(
            dataset_path=temp_file_path,
            target_col=target_column,
            task_hint=task_hint if task_hint != "Auto-detect" else None
        )
        
        progress_bar.progress(100)
        
        if result["success"]:
            status_text.success("✅ Pipeline completed successfully!")
            display_results(result)
        else:
            status_text.error(f"❌ Pipeline failed: {result['error']}")
            st.error("Please check the logs for more details.")
    
    except Exception as e:
        status_text.error(f"❌ Unexpected error: {e}")
        st.exception(e)
    
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def display_results(result):
    """Display pipeline results."""
    st.header("📊 Results")
    
    model_artifact = result['model_artifact']
    deployment_info = result['deployment_info']
    
    # Model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 Model Information</h3>
            <p><strong>Algorithm:</strong> {model_artifact.model_type}</p>
            <p><strong>Task:</strong> {model_artifact.task_type}</p>
            <p><strong>Target:</strong> {model_artifact.target_column}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Performance Metrics</h3>
        """, unsafe_allow_html=True)
        
        for metric, value in model_artifact.metrics.items():
            st.write(f"**{metric.replace('_', ' ').title()}:** {value:.4f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚀 Deployment</h3>
            <p><strong>Status:</strong> {deployment_info.status}</p>
            <p><strong>App Path:</strong> {deployment_info.app_path}</p>
            <p><strong>URL:</strong> {deployment_info.app_url}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Generated artifacts
    st.header("📦 Generated Artifacts")
    
    artifacts = result['artifacts']
    for artifact_type, artifact_path in artifacts.items():
        st.write(f"**{artifact_type.replace('_', ' ').title()}:** `{artifact_path}`")
    
    # Launch app button
    st.header("🎮 Test Your Model")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Generated App", type="primary", use_container_width=True):
            try:
                import subprocess
                subprocess.Popen(["streamlit", "run", deployment_info.app_path])
                st.success("🎉 App launched! Check your terminal for the URL.")
            except Exception as e:
                st.error(f"❌ Could not launch app: {e}")
    
    # Instructions
    st.info("""
    💡 **Next Steps:**
    1. Click "Launch Generated App" to test your model
    2. Use the prediction interface to make predictions
    3. Review the model performance and metrics
    4. Download the generated artifacts for future use
    """)


if __name__ == "__main__":
    main()
