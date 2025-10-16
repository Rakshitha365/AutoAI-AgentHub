import os
import sys
import subprocess
import json
from pathlib import Path


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def setup_directories():
    """Setup project directory structure."""
    print("Setting up directories...")
    
    directories = [
        "artifacts",
        "artifacts/models",
        "artifacts/logs",
        "artifacts/reports",
        "artifacts/streamlit"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created {directory}")
    
    print("Directory structure created")


def generate_sample_data():
    """Generate sample datasets."""
    print("Generating sample datasets...")
    try:
        subprocess.run([sys.executable, "data/generate_datasets.py"], check=True)
        print("Sample datasets generated")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating sample data: {e}")
        return False


def run_tests():
    """Run test suite."""
    print("Running tests...")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        print("All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Tests failed: {e}")
        return False


def main():
    """Main deployment function."""
    print("AutoAI AgentHub - Deployment")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found. Please run from AutoAI-AgentHub directory")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Generate sample data
    if not generate_sample_data():
        print("Warning: Could not generate sample data")
    
    # Run tests
    if not run_tests():
        print("Warning: Some tests failed")
    
    print("\nDeployment completed successfully!")
    print("\nNext Steps:")
    print("1. Launch web interface: python main.py --web")
    print("2. Upload your own data and generate models")
    print("\nFor help, run: python main.py --help")


if __name__ == "__main__":
    main()
