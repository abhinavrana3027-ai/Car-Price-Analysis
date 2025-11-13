#!/usr/bin/env python3
"""
Car Price Prediction - Complete Machine Learning Pipeline
Author: Abhinav Rana
Date: November 2025

This script demonstrates a complete end-to-end machine learning pipeline for 
predicting car prices using multiple regression algorithms.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class CarPricePredictor:
    """
    Complete Car Price Prediction Pipeline
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        return self
    
    def perform_eda(self):
        """Exploratory Data Analysis"""
        print("\nPerforming EDA...")
        
        # Price distribution
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['Price'], kde=True, bins=50)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        
        # Price by Brand
        plt.subplot(1, 2, 2)
        top_brands = self.df['Brand'].value_counts().head(10).index
        sns.boxplot(data=self.df[self.df['Brand'].isin(top_brands)], 
                   x='Brand', y='Price')
        plt.xticks(rotation=45)
        plt.title('Price by Top 10 Brands')
        plt.tight_layout()
        plt.savefig('eda_price_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: eda_price_distribution.png")
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation = self.df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: correlation_heatmap.png")
        
        return self
    
    def feature_engineering(self):
        """Create new features"""
        print("\nEngineering features...")
        
        # Calculate age
        current_year = datetime.now().year
        self.df['Age'] = current_year - self.df['Year']
        
        # Mileage per year
        self.df['Mileage_per_Year'] = self.df['Mileage'] / (self.df['Age'] + 1)
        
        # Price categories
        self.df['Price_Category'] = pd.cut(self.df['Price'], 
                                           bins=[0, 20000, 40000, 60000, 100000],
                                           labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
        
        print(f"New features created: Age, Mileage_per_Year, Price_Category")
        print(f"\nUpdated shape: {self.df.shape}")
        
        return self
    
    def preprocess_data(self):
        """Preprocess data for modeling"""
        print("\nPreprocessing data...")
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col])
        
        # Select features for modeling
        feature_cols = ['Year', 'Engine Size', 'Mileage', 'Age', 'Mileage_per_Year',
                       'Brand_Encoded', 'Fuel Type_Encoded', 'Transmission_Encoded', 
                       'Condition_Encoded']
        
        X = self.df[feature_cols]
        y = self.df['Price']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self
    
    def train_models(self):
        """Train multiple models"""
        print("\nTraining models...")
        
        # Define models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Metrics
            self.results[name] = {
                'train_r2': r2_score(self.y_train, y_pred_train),
                'test_r2': r2_score(self.y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'mae': mean_absolute_error(self.y_test, y_pred_test)
            }
            
            print(f"{name} - Test R2: {self.results[name]['test_r2']:.4f}, "
                  f"RMSE: ${self.results[name]['rmse']:.2f}, "
                  f"MAE: ${self.results[name]['mae']:.2f}")
        
        return self
    
    def visualize_results(self):
        """Visualize model performance"""
        print("\nCreating visualizations...")
        
        # Model comparison
        results_df = pd.DataFrame(self.results).T
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        results_df[['train_r2', 'test_r2']].plot(kind='bar', ax=plt.gca())
        plt.title('Model R² Scores Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['Train R²', 'Test R²'])
        plt.grid(axis='y')
        
        plt.subplot(1, 2, 2)
        results_df[['rmse', 'mae']].plot(kind='bar', ax=plt.gca())
        plt.title('Model Error Metrics')
        plt.ylabel('Error ($)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['RMSE', 'MAE'])
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: model_comparison.png")
        
        # Best model predictions
        best_model_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Actual vs Predicted Prices - {best_model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print("Saved: actual_vs_predicted.png")
        
        return self
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("FINAL MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('test_r2', ascending=False)
        
        print("\nModel Rankings (by Test R² Score):")
        print(results_df.to_string())
        
        best_model = results_df.index[0]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   - Test R² Score: {results_df.loc[best_model, 'test_r2']:.4f}")
        print(f"   - RMSE: ${results_df.loc[best_model, 'rmse']:.2f}")
        print(f"   - MAE: ${results_df.loc[best_model, 'mae']:.2f}")
        
        print("\n" + "="*60)
        
        return self


def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("CAR PRICE PREDICTION - ML PIPELINE")
    print("Author: Abhinav Rana")
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = CarPricePredictor('car_price_prediction_.csv')
    
    # Execute pipeline
    (pipeline
     .load_data()
     .perform_eda()
     .feature_engineering()
     .preprocess_data()
     .train_models()
     .visualize_results()
     .print_summary())
    
    print("\n✅ Pipeline completed successfully!")
    print("\nGenerated files:")
    print("  - eda_price_distribution.png")
    print("  - correlation_heatmap.png")
    print("  - model_comparison.png")
    print("  - actual_vs_predicted.png")
    

if __name__ == "__main__":
    main()
