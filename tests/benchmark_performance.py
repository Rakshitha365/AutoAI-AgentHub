import time
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from src.agents.data_agent import DataAgent
from src.agents.model_agent import ModelAgent
from src.agents.deploy_agent import DeployAgent


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for AutoAI framework."""
    
    def __init__(self, config_path: str = "src/config/config.json"):
        """Initialize performance benchmark."""
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for benchmarks."""
        logger = logging.getLogger('benchmark')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def generate_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate test datasets of various sizes."""
        self.logger.info("Generating test datasets...")
        
        datasets = {}
        
        # Small dataset (100 rows, 5 features)
        np.random.seed(42)
        small_data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'feature4': np.random.uniform(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        }
        datasets['small_classification'] = pd.DataFrame(small_data)
        
        # Medium dataset (500 rows, 8 features)
        medium_data = {
            'feature1': np.random.normal(0, 1, 500),
            'feature2': np.random.normal(0, 1, 500),
            'feature3': np.random.choice(['A', 'B', 'C', 'D'], 500),
            'feature4': np.random.uniform(0, 1, 500),
            'feature5': np.random.exponential(1, 500),
            'feature6': np.random.poisson(3, 500),
            'feature7': np.random.choice(['X', 'Y', 'Z'], 500),
            'target': np.random.choice([0, 1], 500)
        }
        datasets['medium_classification'] = pd.DataFrame(medium_data)
        
        # Large dataset (1000 rows, 10 features)
        large_data = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'feature3': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000),
            'feature4': np.random.uniform(0, 1, 1000),
            'feature5': np.random.exponential(1, 1000),
            'feature6': np.random.poisson(3, 1000),
            'feature7': np.random.choice(['X', 'Y', 'Z'], 1000),
            'feature8': np.random.normal(5, 2, 1000),
            'feature9': np.random.choice(['P', 'Q', 'R'], 1000),
            'target': np.random.choice([0, 1], 1000)
        }
        datasets['large_classification'] = pd.DataFrame(large_data)
        
        # Regression datasets
        datasets['small_regression'] = datasets['small_classification'].copy()
        datasets['small_regression']['target'] = np.random.normal(0, 1, 100)
        
        datasets['medium_regression'] = datasets['medium_classification'].copy()
        datasets['medium_regression']['target'] = np.random.normal(0, 1, 500)
        
        datasets['large_regression'] = datasets['large_classification'].copy()
        datasets['large_regression']['target'] = np.random.normal(0, 1, 1000)
        
        self.logger.info(f"Generated {len(datasets)} test datasets")
        return datasets
    
    def benchmark_data_agent(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Benchmark Data Agent performance."""
        self.logger.info("Benchmarking Data Agent...")
        
        config = {
            "max_file_size_mb": 10,
            "default_test_size": 0.2,
            "random_state": 42,
            "max_categorical_cardinality": 50
        }
        
        logger = logging.getLogger('data_agent_benchmark')
        data_agent = DataAgent(config, logger)
        
        results = {}
        
        for dataset_name, df in datasets.items():
            self.logger.info(f"Testing {dataset_name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            # Benchmark analyze
            start_time = time.time()
            metadata = data_agent.analyze(df)
            analyze_time = time.time() - start_time
            
            # Benchmark preprocess
            start_time = time.time()
            processed_payload = data_agent.preprocess(df, 'target')
            preprocess_time = time.time() - start_time
            
            # Benchmark quality validation
            start_time = time.time()
            quality_report = data_agent.validate_data_quality(df)
            quality_time = time.time() - start_time
            
            results[dataset_name] = {
                'dataset_size': df.shape,
                'analyze_time': analyze_time,
                'preprocess_time': preprocess_time,
                'quality_time': quality_time,
                'total_time': analyze_time + preprocess_time + quality_time,
                'quality_score': quality_report['quality_score'],
                'features_processed': processed_payload.X_train.shape[1]
            }
        
        return results
    
    def benchmark_model_agent(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Benchmark Model Agent performance."""
        self.logger.info("Benchmarking Model Agent...")
        
        config = {
            "model_trials": {
                "classification": ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier"],
                "regression": ["LinearRegression", "Ridge", "RandomForestRegressor"]
            },
            "random_state": 42
        }
        
        logger = logging.getLogger('model_agent_benchmark')
        model_agent = ModelAgent(config, logger)
        
        results = {}
        
        for dataset_name, df in datasets.items():
            self.logger.info(f"Testing {dataset_name}")
            
            # Prepare data
            data_agent = DataAgent(config, logger)
            metadata = data_agent.analyze(df)
            processed_payload = data_agent.preprocess(df, 'target')
            
            # Benchmark model training
            start_time = time.time()
            model_artifact = model_agent.train_and_evaluate(processed_payload)
            training_time = time.time() - start_time
            
            # Benchmark prediction
            start_time = time.time()
            model = model_agent.load_model(model_artifact.model_path)
            predictions = model_agent.predict(model, processed_payload.X_test.head(10))
            prediction_time = time.time() - start_time
            
            results[dataset_name] = {
                'dataset_size': df.shape,
                'task_type': model_artifact.task_type,
                'model_type': model_artifact.model_type,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'metrics': model_artifact.metrics,
                'best_metric': max(model_artifact.metrics.values()) if model_artifact.task_type == 'classification' else min(model_artifact.metrics.values())
            }
        
        return results
    
    def benchmark_end_to_end_pipeline(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Benchmark complete end-to-end pipeline."""
        self.logger.info("Benchmarking end-to-end pipeline...")
        
        orchestrator = Orchestrator(self.config_path)
        results = {}
        
        for dataset_name, df in datasets.items():
            self.logger.info(f"Testing {dataset_name}")
            
            # Save dataset to temporary file
            temp_file = f"temp_{dataset_name}.csv"
            df.to_csv(temp_file, index=False)
            
            try:
                # Benchmark complete pipeline
                start_time = time.time()
                result = orchestrator.run_pipeline(
                    dataset_path=temp_file,
                    target_col='target'
                )
                total_time = time.time() - start_time
                
                if result['success']:
                    results[dataset_name] = {
                        'dataset_size': df.shape,
                        'total_time': total_time,
                        'success': True,
                        'model_type': result['model_artifact'].model_type,
                        'task_type': result['model_artifact'].task_type,
                        'metrics': result['model_artifact'].metrics,
                        'artifacts_generated': len(result['artifacts'])
                    }
                else:
                    results[dataset_name] = {
                        'dataset_size': df.shape,
                        'total_time': total_time,
                        'success': False,
                        'error': result['error']
                    }
            
            finally:
                # Cleanup
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
        
        return results
    
    def benchmark_memory_usage(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Benchmark memory usage during processing."""
        self.logger.info("Benchmarking memory usage...")
        
        import psutil
        import os
        
        results = {}
        process = psutil.Process(os.getpid())
        
        for dataset_name, df in datasets.items():
            self.logger.info(f"Testing {dataset_name}")
            
            # Measure baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process dataset
            orchestrator = Orchestrator(self.config_path)
            temp_file = f"temp_{dataset_name}.csv"
            df.to_csv(temp_file, index=False)
            
            try:
                result = orchestrator.run_pipeline(temp_file, 'target')
                
                # Measure peak memory
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = peak_memory - baseline_memory
                
                results[dataset_name] = {
                    'dataset_size': df.shape,
                    'baseline_memory_mb': baseline_memory,
                    'peak_memory_mb': peak_memory,
                    'memory_usage_mb': memory_usage,
                    'memory_per_row_mb': memory_usage / df.shape[0] if df.shape[0] > 0 else 0
                }
            
            finally:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        self.logger.info("Starting comprehensive performance benchmark...")
        
        # Generate test datasets
        datasets = self.generate_test_datasets()
        
        # Run benchmarks
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'data_agent': self.benchmark_data_agent(datasets),
            'model_agent': self.benchmark_model_agent(datasets),
            'end_to_end': self.benchmark_end_to_end_pipeline(datasets),
            'memory_usage': self.benchmark_memory_usage(datasets)
        }
        
        # Calculate summary statistics
        benchmark_results['summary'] = self._calculate_summary_stats(benchmark_results)
        
        # Save results
        self._save_results(benchmark_results)
        
        return benchmark_results
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark results."""
        summary = {}
        
        # Data Agent summary
        data_times = [r['total_time'] for r in results['data_agent'].values()]
        summary['data_agent'] = {
            'avg_time': np.mean(data_times),
            'min_time': np.min(data_times),
            'max_time': np.max(data_times),
            'std_time': np.std(data_times)
        }
        
        # Model Agent summary
        model_times = [r['training_time'] for r in results['model_agent'].values()]
        summary['model_agent'] = {
            'avg_training_time': np.mean(model_times),
            'min_training_time': np.min(model_times),
            'max_training_time': np.max(model_times),
            'std_training_time': np.std(model_times)
        }
        
        # End-to-end summary
        e2e_times = [r['total_time'] for r in results['end_to_end'].values() if r['success']]
        summary['end_to_end'] = {
            'avg_time': np.mean(e2e_times) if e2e_times else 0,
            'min_time': np.min(e2e_times) if e2e_times else 0,
            'max_time': np.max(e2e_times) if e2e_times else 0,
            'std_time': np.std(e2e_times) if e2e_times else 0,
            'success_rate': len(e2e_times) / len(results['end_to_end'])
        }
        
        # Memory usage summary
        memory_usage = [r['memory_usage_mb'] for r in results['memory_usage'].values()]
        summary['memory'] = {
            'avg_usage_mb': np.mean(memory_usage),
            'min_usage_mb': np.min(memory_usage),
            'max_usage_mb': np.max(memory_usage),
            'std_usage_mb': np.std(memory_usage)
        }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to: {results_file}")
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable performance report."""
        report = []
        report.append("# AutoAI AgentHub - Performance Benchmark Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        summary = results['summary']
        
        report.append(f"- **Average End-to-End Time**: {summary['end_to_end']['avg_time']:.2f} seconds")
        report.append(f"- **Success Rate**: {summary['end_to_end']['success_rate']:.1%}")
        report.append(f"- **Average Memory Usage**: {summary['memory']['avg_usage_mb']:.1f} MB")
        report.append("")
        
        # Data Agent Performance
        report.append("## Data Agent Performance")
        data_summary = summary['data_agent']
        report.append(f"- **Average Processing Time**: {data_summary['avg_time']:.2f} seconds")
        report.append(f"- **Time Range**: {data_summary['min_time']:.2f} - {data_summary['max_time']:.2f} seconds")
        report.append("")
        
        # Model Agent Performance
        report.append("## Model Agent Performance")
        model_summary = summary['model_agent']
        report.append(f"- **Average Training Time**: {model_summary['avg_training_time']:.2f} seconds")
        report.append(f"- **Training Time Range**: {model_summary['min_training_time']:.2f} - {model_summary['max_training_time']:.2f} seconds")
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        
        for dataset_name, result in results['end_to_end'].items():
            if result['success']:
                report.append(f"### {dataset_name}")
                report.append(f"- **Dataset Size**: {result['dataset_size'][0]} rows, {result['dataset_size'][1]} columns")
                report.append(f"- **Total Time**: {result['total_time']:.2f} seconds")
                report.append(f"- **Model Type**: {result['model_type']}")
                report.append(f"- **Task Type**: {result['task_type']}")
                report.append(f"- **Artifacts Generated**: {result['artifacts_generated']}")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main benchmark function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoAI Performance Benchmark")
    parser.add_argument(
        "--config",
        default="src/config/config.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output",
        help="Output file for results"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate performance report"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = PerformanceBenchmark(args.config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate report if requested
    if args.report:
        report = benchmark.generate_performance_report(results)
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
    
    print("🎉 Performance benchmark completed!")
    print(f"Average end-to-end time: {results['summary']['end_to_end']['avg_time']:.2f} seconds")
    print(f"Success rate: {results['summary']['end_to_end']['success_rate']:.1%}")


if __name__ == "__main__":
    main()
