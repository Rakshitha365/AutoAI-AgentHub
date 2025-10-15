import logging
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from src.core.dataclasses import ProcessedPayload, ModelArtifact, DeploymentInfo


class Orchestrator:
    """Main orchestrator that coordinates all agents in the AI automation pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize the Orchestrator.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Initialize agents (import here to avoid circular imports)
        from src.agents.data_agent import DataAgent
        from src.agents.model_agent import ModelAgent
        from src.agents.deploy_agent import DeployAgent
        
        self.data_agent = DataAgent(self.config, self.logger)
        self.model_agent = ModelAgent(self.config, self.logger)
        self.deploy_agent = DeployAgent(self.config, self.logger)
        
        self.logger.info("Orchestrator: Initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # Use default configuration if file not found
            return self._get_default_config()
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "artifact_dir": "artifacts",
            "models_dir": "artifacts/models",
            "logs_dir": "artifacts/logs",
            "reports_dir": "artifacts/reports",
            "max_file_size_mb": 10,
            "default_test_size": 0.2,
            "random_state": 42,
            "max_categorical_cardinality": 50,
            "streamlit_port": 8501,
            "log_level": "INFO",
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
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("Orchestrator")
        logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        logs_dir = self.config.get("logs_dir", "artifacts/logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, f"orchestrator_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def run_pipeline(self, dataset_path: str, target_col: Optional[str] = None, 
                    task_hint: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete AI automation pipeline.
        
        Args:
            dataset_path: Path to CSV dataset file
            target_col: Optional target column name
            task_hint: Optional task type hint (classification/regression)
            
        Returns:
            Dictionary containing pipeline results
        """
        self.logger.info(f"Orchestrator: Starting pipeline for dataset: {dataset_path}")
        
        run_id = datetime.now().strftime('%Y%m%d%H%M%S')
        artifacts = {}
        
        try:
            # Step 1: Data Loading and Analysis
            self.logger.info("Orchestrator: Step 1 - Loading and analyzing data")
            df = self.data_agent.load(dataset_path)
            metadata = self.data_agent.analyze(df)
            
            # Auto-detect target column if not provided
            if not target_col:
                if metadata["target_candidates"]:
                    target_col = metadata["target_candidates"][0]
                    self.logger.info(f"Orchestrator: Auto-detected target column: {target_col}")
                else:
                    # Use last column as fallback
                    target_col = df.columns[-1]
                    self.logger.info(f"Orchestrator: Using last column as target: {target_col}")
            
            # Auto-detect task type if not provided
            if not task_hint:
                if df[target_col].nunique() < len(df) * 0.1:
                    task_hint = "classification"
                else:
                    task_hint = "regression"
                self.logger.info(f"Orchestrator: Auto-detected task type: {task_hint}")
            
            # Update metadata
            metadata["target_column"] = target_col
            metadata["task_type"] = task_hint
            
            # Step 2: Data Preprocessing
            self.logger.info("Orchestrator: Step 2 - Preprocessing data")
            processed_payload = self.data_agent.preprocess(df, target_col)
            artifacts["preprocessor"] = processed_payload.preprocessor_path
            
            # Step 3: Model Training and Evaluation
            self.logger.info("Orchestrator: Step 3 - Training and evaluating models")
            model_artifact = self.model_agent.train_and_evaluate(processed_payload)
            artifacts["model"] = model_artifact.model_path
            
            # Save metrics to file
            metrics_path = os.path.join(self.config["reports_dir"], f"metrics_{run_id}.json")
            self.model_agent.save_model_metrics(model_artifact.metrics, metrics_path)
            artifacts["metrics"] = metrics_path
            
            # Step 4: Model Deployment
            self.logger.info("Orchestrator: Step 4 - Generating deployment interface")
            streamlit_script_path = self.deploy_agent.generate_streamlit_script(model_artifact)
            artifacts["streamlit_app"] = streamlit_script_path
            
            # Optional: Create API
            try:
                api_script_path = self.deploy_agent.create_prediction_api(model_artifact)
                artifacts["api"] = api_script_path
            except Exception as e:
                self.logger.warning(f"Orchestrator: API generation failed: {str(e)}")
            
            # Step 5: Generate Reports
            self.logger.info("Orchestrator: Step 5 - Generating reports")
            report_path = self._generate_summary_report(
                run_id, dataset_path, model_artifact, artifacts
            )
            artifacts["summary"] = report_path
            
            # Create deployment info
            deployment_info = DeploymentInfo(
                app_path=streamlit_script_path,
                app_url=f"http://localhost:{self.config.get('streamlit_port', 8501)}",
                model_artifact=model_artifact,
                deployment_timestamp=datetime.now().isoformat(),
                status="ready"
            )
            
            self.logger.info("Orchestrator: Pipeline completed successfully")
            
            return {
                "success": True,
                "run_id": run_id,
                "model_artifact": model_artifact,
                "deployment_info": deployment_info,
                "artifacts": artifacts,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Orchestrator: Pipeline failed: {str(e)}")
            return {
                "success": False,
                "run_id": run_id,
                "error": str(e),
                "artifacts": artifacts
            }
    
    def _generate_summary_report(self, run_id: str, dataset_path: str, 
                               model_artifact: ModelArtifact, 
                               artifacts: Dict[str, str]) -> str:
        """Generate a summary report of the pipeline run."""
        reports_dir = self.config.get("reports_dir", "artifacts/reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, f"summary_report_{run_id}.json")
        
        report = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "model_type": model_artifact.model_type,
            "task_type": model_artifact.task_type,
            "target_column": model_artifact.target_column,
            "metrics": model_artifact.metrics,
            "artifacts": artifacts,
            "feature_count": len(model_artifact.feature_names),
            "status": "completed"
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Orchestrator: Summary report saved to {report_path}")
        return report_path
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "status": "ready",
            "agents_initialized": {
                "data_agent": self.data_agent is not None,
                "model_agent": self.model_agent is not None,
                "deploy_agent": self.deploy_agent is not None
            },
            "config_loaded": self.config is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup_artifacts(self, run_id: Optional[str] = None) -> None:
        """Clean up artifacts for a specific run."""
        if run_id:
            # Clean up specific run artifacts
            artifact_patterns = [
                f"*{run_id}*",
                f"preprocessor_{run_id}*",
                f"best_model_{run_id}*",
                f"streamlit_app_{run_id}*",
                f"summary_report_{run_id}*"
            ]
            
            for pattern in artifact_patterns:
                # Implementation would depend on specific cleanup requirements
                self.logger.info(f"Orchestrator: Cleanup pattern {pattern} for run {run_id}")
        else:
            self.logger.info("Orchestrator: Cleanup all artifacts")
