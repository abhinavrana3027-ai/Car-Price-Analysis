"""Model Evaluation and Comparison Utilities

Author: Abhinav Rana
Date: November 2025

This module provides comprehensive model evaluation tools including
metric calculations, cross-validation, and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and comparison tools."""
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.results = {}
        self.models = {}
        
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            model: Trained model object
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'predictions': y_test_pred
        }
        
        # Store results
        self.results[model_name] = metrics
        self.models[model_name] = model
        
        return metrics
    
    def calculate_mape(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: MAPE value
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def cross_validate_model(self, model, X, y, cv=5):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        print(f"\nCross-Validation Results (CV={cv}):")
        print(f"  Mean R² Score: {cv_results['mean_score']:.4f}")
        print(f"  Std Deviation: {cv_results['std_score']:.4f}")
        
        return cv_results
    
    def compare_models(self):
        """
        Compare all evaluated models and return sorted results.
        
        Returns:
            pd.DataFrame: Comparison dataframe sorted by test R²
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return None
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Train RMSE': metrics['train_rmse'],
                'Test RMSE': metrics['test_rmse'],
                'Train MAE': metrics['train_mae'],
                'Test MAE': metrics['test_mae']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test R²', ascending=False).reset_index(drop=True)
        
        return df
    
    def print_detailed_results(self):
        """
        Print detailed results for all evaluated models.
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        comparison_df = self.compare_models()
        
        if comparison_df is not None:
            print("\n" + comparison_df.to_string(index=False))
            
            # Best model
            best_model = comparison_df.iloc[0]['Model']
            best_r2 = comparison_df.iloc[0]['Test R²']
            
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   Test R² Score: {best_r2:.4f}")
            print(f"   This model explains {best_r2*100:.2f}% of the variance in car prices.")
    
    def plot_predictions(self, model_name, y_test, figsize=(10, 6)):
        """
        Plot actual vs predicted values for a specific model.
        
        Args:
            model_name (str): Name of the model
            y_test: True test values
            figsize (tuple): Figure size
        """
        if model_name not in self.results:
            print(f"Model '{model_name}' not found in results.")
            return
        
        y_pred = self.results[model_name]['predictions']
        r2 = self.results[model_name]['test_r2']
        
        plt.figure(figsize=figsize)
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title(f'{model_name} - Actual vs Predicted\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt
    
    def plot_residuals(self, model_name, y_test, figsize=(10, 6)):
        """
        Plot residuals for a specific model.
        
        Args:
            model_name (str): Name of the model
            y_test: True test values
            figsize (tuple): Figure size
        """
        if model_name not in self.results:
            print(f"Model '{model_name}' not found in results.")
            return
        
        y_pred = self.results[model_name]['predictions']
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price ($)', fontsize=11)
        axes[0].set_ylabel('Residuals ($)', fontsize=11)
        axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals ($)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return plt
    
    def plot_learning_curve(self, model, X, y, model_name, cv=5, figsize=(10, 6)):
        """
        Plot learning curve to diagnose bias/variance.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            model_name (str): Name of the model
            cv (int): Cross-validation folds
            figsize (tuple): Figure size
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        plt.plot(train_sizes, val_mean, label='Validation Score', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt
    
    def get_best_model(self):
        """
        Get the best performing model based on test R².
        
        Returns:
            tuple: (model_name, model_object, metrics)
        """
        if not self.results:
            return None
        
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        return best_name, self.models[best_name], self.results[best_name]


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Quick function to evaluate a regression model.
    
    Args:
        model: Trained model
        X_train, X_test: Train and test features
        y_train, y_test: Train and test targets
        model_name (str): Name of the model
        
    Returns:
        dict: Evaluation metrics
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}")
    print(f"Training R² Score:    {metrics['train_r2']:.4f}")
    print(f"Test R² Score:        {metrics['test_r2']:.4f}")
    print(f"Training RMSE:        ${metrics['train_rmse']:,.2f}")
    print(f"Test RMSE:            ${metrics['test_rmse']:,.2f}")
    print(f"Training MAE:         ${metrics['train_mae']:,.2f}")
    print(f"Test MAE:             ${metrics['test_mae']:,.2f}")
    print(f"{'='*60}")
    
    return metrics


if __name__ == "__main__":
    print("Model Evaluation Utilities Module")
    print("Import this module to use ModelEvaluator class and evaluation functions.")
