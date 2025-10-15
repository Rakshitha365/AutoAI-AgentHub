import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing, load_wine, load_breast_cancer
from sklearn.datasets import make_classification, make_regression
import os


def generate_iris_dataset():
    """Generate Iris dataset for classification testing."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df


def generate_california_housing_dataset():
    """Generate California Housing dataset for regression testing."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MEDV'] = housing.target
    return df


def generate_wine_dataset():
    """Generate Wine dataset for classification testing."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['wine_type'] = wine.target_names[wine.target]
    return df


def generate_breast_cancer_dataset():
    """Generate Breast Cancer dataset for binary classification testing."""
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = cancer.target_names[cancer.target]
    return df


def generate_custom_classification_dataset(n_samples=200, n_features=5):
    """Generate custom classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


def generate_custom_regression_dataset(n_samples=200, n_features=5):
    """Generate custom regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


def generate_customer_churn_dataset():
    """Generate synthetic customer churn dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'tenure': np.random.exponential(2, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation between features and churn
    df.loc[df['contract_type'] == 'Month-to-month', 'churn'] = np.random.choice([0, 1], 
        size=len(df[df['contract_type'] == 'Month-to-month']), p=[0.4, 0.6])
    
    return df


def generate_sales_prediction_dataset():
    """Generate synthetic sales prediction dataset."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'advertising_spend': np.random.exponential(1000, n_samples),
        'social_media_followers': np.random.poisson(10000, n_samples),
        'season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'holiday': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'competitor_price': np.random.normal(50, 10, n_samples),
        'sales': np.random.normal(1000, 200, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    df['sales'] = (df['advertising_spend'] * 0.5 + 
                   df['social_media_followers'] * 0.01 + 
                   df['holiday'] * 200 + 
                   np.random.normal(0, 50, n_samples))
    
    return df


def save_all_datasets():
    """Generate and save all sample datasets."""
    datasets = {
        'iris.csv': generate_iris_dataset(),
        'california_housing.csv': generate_california_housing_dataset(),
        'wine.csv': generate_wine_dataset(),
        'breast_cancer.csv': generate_breast_cancer_dataset(),
        'custom_classification.csv': generate_custom_classification_dataset(),
        'custom_regression.csv': generate_custom_regression_dataset(),
        'customer_churn.csv': generate_customer_churn_dataset(),
        'sales_prediction.csv': generate_sales_prediction_dataset()
    }
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    for filename, df in datasets.items():
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Generated {filename}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    save_all_datasets()
