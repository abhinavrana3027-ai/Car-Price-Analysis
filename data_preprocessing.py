"""Data Preprocessing Utilities for Car Price Prediction

Author: Abhinav Rana
Date: November 2025

This module provides reusable data preprocessing functions for the car price
prediction project, including feature engineering, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data preprocessing operations for car price prediction."""
    
    def __init__(self):
        """Initialize the DataPreprocessor with encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_validate_data(self, filepath):
        """
        Load data from CSV and perform basic validation.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Basic validation
        print(f"\nDataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Handle missing values if any
        if df.isnull().sum().sum() > 0:
            print("\nHandling missing values...")
            df = df.dropna()
            print(f"Shape after handling missing values: {df.shape}")
            
        return df
    
    def create_age_feature(self, df):
        """
        Create 'Age' feature from 'Year' column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Age feature
        """
        current_year = datetime.now().year
        df['Age'] = current_year - df['Year']
        print(f"Created 'Age' feature (range: {df['Age'].min()} to {df['Age'].max()} years)")
        return df
    
    def create_mileage_per_year(self, df):
        """
        Create 'Mileage_per_Year' feature.
        
        Args:
            df (pd.DataFrame): Input dataframe with Age and Mileage
            
        Returns:
            pd.DataFrame: Dataframe with Mileage_per_Year feature
        """
        df['Mileage_per_Year'] = df['Mileage'] / (df['Age'] + 1)  # +1 to avoid division by zero
        print(f"Created 'Mileage_per_Year' feature (avg: {df['Mileage_per_Year'].mean():.2f})")
        return df
    
    def create_price_category(self, df):
        """
        Create 'Price_Category' based on price quartiles.
        
        Args:
            df (pd.DataFrame): Input dataframe with Price
            
        Returns:
            pd.DataFrame: Dataframe with Price_Category feature
        """
        df['Price_Category'] = pd.qcut(df['Price'], q=4, labels=['Budget', 'Economy', 'Mid-Range', 'Premium'])
        print(f"Created 'Price_Category' feature")
        print(f"Distribution:\n{df['Price_Category'].value_counts()}")
        return df
    
    def encode_categorical_features(self, df, categorical_columns):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_columns (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        print(f"\nEncoding categorical features: {categorical_columns}")
        
        for col in categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_Encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                print(f"  {col}: {df[col].nunique()} unique values encoded")
        
        return df
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            
        Returns:
            tuple: Scaled training and test features
        """
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_importance_info(self, df):
        """
        Get basic feature statistics for understanding data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary containing feature statistics
        """
        stats = {
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict()
        }
        
        print("\nFeature Information:")
        print(f"  Numeric features ({len(stats['numeric_features'])}): {stats['numeric_features']}")
        print(f"  Categorical features ({len(stats['categorical_features'])}): {stats['categorical_features']}")
        
        return stats
    
    def detect_outliers(self, df, column, method='IQR'):
        """
        Detect outliers in a numeric column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to check for outliers
            method (str): Method for outlier detection ('IQR' or 'zscore')
            
        Returns:
            pd.Series: Boolean series indicating outliers
        """
        if method == 'IQR':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        else:  # zscore method
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > 3
        
        print(f"\nOutliers in '{column}': {outliers.sum()} ({outliers.sum()/len(df)*100:.2f}%)")
        return outliers


def preprocess_pipeline(filepath, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for car price data.
    
    Args:
        filepath (str): Path to the CSV file
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    from sklearn.model_selection import train_test_split
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and validate data
    df = preprocessor.load_and_validate_data(filepath)
    
    # Feature engineering
    df = preprocessor.create_age_feature(df)
    df = preprocessor.create_mileage_per_year(df)
    df = preprocessor.create_price_category(df)
    
    # Encode categorical features
    categorical_cols = ['Brand', 'Fuel_Type', 'Transmission', 'Condition', 'Model', 'Price_Category']
    df = preprocessor.encode_categorical_features(df, categorical_cols)
    
    # Select features for modeling
    feature_cols = ['Age', 'Engine_Size', 'Mileage', 'Mileage_per_Year',
                    'Brand_Encoded', 'Fuel_Type_Encoded', 'Transmission_Encoded',
                    'Condition_Encoded', 'Model_Encoded']
    
    X = df[feature_cols]
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Run preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        'car_price_prediction_.csv'
    )
    
    print(f"\n📊 Final Dataset Summary:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Target range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
