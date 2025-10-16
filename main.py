import argparse
import json
import os
import sys
from pathlib import Path
from src.core.orchestrator import Orchestrator


def main():
    """Main entry point for the AutoAI AgentHub."""
    parser = argparse.ArgumentParser(
        description="AutoAI AgentHub - Automated ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with dataset and auto-detect target
  python main.py --dataset data/iris.csv
  
  # Run with specific target column
  python main.py --dataset data/housing.csv --target price
  
  # Run with task type hint
  python main.py --dataset data/custom.csv --target outcome --task classification
  
  # Launch web interface
  python main.py --web
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the input dataset (CSV file)"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Name of the target column in the dataset"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        help="Hint for the ML task type"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.json",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch Streamlit web interface"
    )
    
    args = parser.parse_args()
    
    # Launch web interface if requested
    if args.web:
        launch_web_interface()
        return
    
    # Validate required arguments for CLI mode
    if not args.dataset:
        print("Error: --dataset is required for CLI mode")
        print("Use --web to launch the web interface")
        sys.exit(1)
    
    # Validate configuration file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Update configuration with command line arguments
    if args.output_dir:
        config["artifact_dir"] = args.output_dir
    if args.log_level:
        config["log_level"] = args.log_level
    
    # Initialize orchestrator
    try:
        orchestrator = Orchestrator(args.config)
    except Exception as e:
        print(f"Error initializing orchestrator: {e}")
        sys.exit(1)
    
    # Run pipeline
    print("AutoAI AgentHub - Starting Pipeline")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    if args.target:
        print(f"Target: {args.target}")
    if args.task:
        print(f"Task: {args.task}")
    print("=" * 50)
    
    try:
        result = orchestrator.run_pipeline(
            dataset_path=args.dataset,
            target_col=args.target,
            task_hint=args.task
        )
        
        if result["success"]:
            print("\nPipeline completed successfully!")
            print(f"Run ID: {result['run_id']}")
            print(f"Model: {result['model_artifact'].model_type}")
            print(f"Task: {result['model_artifact'].task_type}")
            print(f"Target: {result['model_artifact'].target_column}")
            
            # Display metrics
            print("\nPerformance Metrics:")
            for metric, value in result['model_artifact'].metrics.items():
                print(f"  • {metric.replace('_', ' ').title()}: {value:.4f}")
            
            # Display artifacts
            print("\nGenerated Artifacts:")
            for artifact_type, artifact_path in result['artifacts'].items():
                print(f"  • {artifact_type}: {artifact_path}")
            
            # Display deployment info
            print(f"\nStreamlit App: {result['deployment_info'].app_path}")
            print(f"Access URL: {result['deployment_info'].app_url}")
            
            # Ask if user wants to launch the app
            try:
                launch = input("\nDo you want to launch the Streamlit app now? (y/n): ").lower()
                if launch in ['y', 'yes']:
                    launch_streamlit_app(result['deployment_info'].app_path)
            except KeyboardInterrupt:
                print("\nGoodbye!")
        else:
            print(f"\nPipeline failed: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


def launch_web_interface():
    """Launch the Streamlit web interface."""
    print("Launching AutoAI Web Interface...")
    print("The interface will open in your default browser")
    print("URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        import subprocess
        subprocess.run(["streamlit", "run", "src/streamlit_app.py"], check=True)
    except FileNotFoundError:
        print("Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWeb interface stopped")
    except Exception as e:
        print(f"Error launching web interface: {e}")
        sys.exit(1)


def launch_streamlit_app(app_path: str):
    """Launch a specific Streamlit app."""
    try:
        import subprocess
        print(f"Launching Streamlit app: {app_path}")
        subprocess.run(["streamlit", "run", app_path], check=True)
    except FileNotFoundError:
        print("Streamlit not found. Please install with: pip install streamlit")
    except Exception as e:
        print(f"Error launching app: {e}")


if __name__ == "__main__":
    main()
