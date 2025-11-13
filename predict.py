"""Car Price Prediction Script

Author: Abhinav Rana
Date: November 2025

This script provides an easy-to-use interface for predicting car prices
using trained machine learning models.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CarPricePredictor:
    """Predict car prices using trained models."""
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to saved model file (optional)
        """
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.label_encoders = model_data.get('encoders', {})
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def preprocess_input(self, car_data):
        """
        Preprocess input data for prediction.
        
        Args:
            car_data (dict): Dictionary containing car features
            
        Returns:
            np.array: Preprocessed feature array
        """
        # Create Age feature
        current_year = datetime.now().year
        age = current_year - car_data.get('Year', current_year)
        
        # Create Mileage_per_Year
        mileage = car_data.get('Mileage', 50000)
        mileage_per_year = mileage / (age + 1)
        
        # Extract features
        features = {
            'Age': age,
            'Engine_Size': car_data.get('Engine_Size', 2.0),
            'Mileage': mileage,
            'Mileage_per_Year': mileage_per_year,
            'Brand': car_data.get('Brand', 'Toyota'),
            'Fuel_Type': car_data.get('Fuel_Type', 'Gasoline'),
            'Transmission': car_data.get('Transmission', 'Automatic'),
            'Condition': car_data.get('Condition', 'Good'),
            'Model': car_data.get('Model', 'Sedan')
        }
        
        return features
    
    def predict_price(self, car_data):
        """
        Predict the price of a car.
        
        Args:
            car_data (dict): Dictionary containing car features
            
        Returns:
            float: Predicted price
        """
        if self.model is None:
            print("Error: No model loaded. Please load a model first.")
            return None
        
        # Preprocess input
        features = self.preprocess_input(car_data)
        
        # Create feature array (order matters!)
        feature_values = [
            features['Age'],
            features['Engine_Size'],
            features['Mileage'],
            features['Mileage_per_Year']
        ]
        
        # Add encoded categorical features
        categorical_features = ['Brand', 'Fuel_Type', 'Transmission', 'Condition', 'Model']
        for cat_feat in categorical_features:
            # Simple encoding (in real scenario, use the actual encoders from training)
            feature_values.append(hash(features[cat_feat]) % 100)
        
        # Convert to numpy array and reshape
        X = np.array(feature_values).reshape(1, -1)
        
        # Scale if scaler is available
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction
    
    def predict_multiple(self, cars_dataframe):
        """
        Predict prices for multiple cars.
        
        Args:
            cars_dataframe (pd.DataFrame): Dataframe containing car features
            
        Returns:
            np.array: Array of predicted prices
        """
        predictions = []
        
        for idx, row in cars_dataframe.iterrows():
            car_data = row.to_dict()
            price = self.predict_price(car_data)
            predictions.append(price)
        
        return np.array(predictions)


def predict_single_car(brand, year, engine_size, fuel_type, transmission, 
                      mileage, condition, model_type):
    """
    Quick function to predict a single car's price.
    
    Args:
        brand (str): Car brand (e.g., 'Toyota', 'Honda')
        year (int): Manufacturing year
        engine_size (float): Engine size in liters
        fuel_type (str): Fuel type ('Gasoline', 'Diesel', 'Electric')
        transmission (str): Transmission type ('Automatic', 'Manual')
        mileage (int): Mileage in kilometers
        condition (str): Condition ('Excellent', 'Good', 'Fair')
        model_type (str): Model type (e.g., 'Sedan', 'SUV')
        
    Returns:
        float: Predicted price
    """
    car_data = {
        'Brand': brand,
        'Year': year,
        'Engine_Size': engine_size,
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Mileage': mileage,
        'Condition': condition,
        'Model': model_type
    }
    
    # For demonstration: use a simple estimation formula
    # In production, this would use the actual trained model
    
    current_year = datetime.now().year
    age = current_year - year
    
    # Base price estimation
    base_price = 30000
    
    # Depreciation
    depreciation = age * 2000
    
    # Mileage penalty
    mileage_penalty = (mileage / 10000) * 500
    
    # Engine size bonus
    engine_bonus = engine_size * 3000
    
    # Condition multiplier
    condition_multipliers = {'Excellent': 1.2, 'Good': 1.0, 'Fair': 0.8}
    condition_mult = condition_multipliers.get(condition, 1.0)
    
    # Transmission bonus
    trans_bonus = 2000 if transmission == 'Automatic' else 0
    
    # Calculate final price
    estimated_price = (base_price - depreciation - mileage_penalty + 
                      engine_bonus + trans_bonus) * condition_mult
    
    return max(estimated_price, 5000)  # Minimum price floor


def demonstrate_prediction():
    """
    Demonstrate the prediction functionality with example cars.
    """
    print("\n" + "="*70)
    print("CAR PRICE PREDICTION DEMONSTRATION")
    print("="*70)
    
    # Example cars
    example_cars = [
        {
            'Brand': 'Toyota',
            'Year': 2020,
            'Engine_Size': 2.5,
            'Fuel_Type': 'Gasoline',
            'Transmission': 'Automatic',
            'Mileage': 30000,
            'Condition': 'Excellent',
            'Model': 'Sedan'
        },
        {
            'Brand': 'Honda',
            'Year': 2018,
            'Engine_Size': 2.0,
            'Fuel_Type': 'Gasoline',
            'Transmission': 'Manual',
            'Mileage': 60000,
            'Condition': 'Good',
            'Model': 'Sedan'
        },
        {
            'Brand': 'BMW',
            'Year': 2021,
            'Engine_Size': 3.0,
            'Fuel_Type': 'Diesel',
            'Transmission': 'Automatic',
            'Mileage': 20000,
            'Condition': 'Excellent',
            'Model': 'SUV'
        }
    ]
    
    print("\n🚗 Predicting prices for example cars...\n")
    
    for i, car in enumerate(example_cars, 1):
        print(f"Car {i}: {car['Brand']} {car['Model']} ({car['Year']})")
        print(f"  Engine: {car['Engine_Size']}L {car['Fuel_Type']}")
        print(f"  Transmission: {car['Transmission']}")
        print(f"  Mileage: {car['Mileage']:,} km")
        print(f"  Condition: {car['Condition']}")
        
        predicted_price = predict_single_car(
            car['Brand'], car['Year'], car['Engine_Size'],
            car['Fuel_Type'], car['Transmission'], car['Mileage'],
            car['Condition'], car['Model']
        )
        
        print(f"  💰 Predicted Price: ${predicted_price:,.2f}")
        print()
    
    print("="*70)
    print("\n✅ Prediction demonstration complete!")
    print("\nNote: These are estimated prices based on a simple formula.")
    print("In production, use the actual trained ML models for accurate predictions.")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_prediction()
    
    print("\n" + "="*70)
    print("HOW TO USE THIS MODULE")
    print("="*70)
    print("""
1. Import the module:
   from predict import CarPricePredictor, predict_single_car

2. For single prediction:
   price = predict_single_car('Toyota', 2020, 2.5, 'Gasoline', 
                               'Automatic', 30000, 'Excellent', 'Sedan')
   print(f'Predicted Price: ${price:,.2f}')

3. For batch predictions with trained model:
   predictor = CarPricePredictor('trained_model.pkl')
   price = predictor.predict_price(car_data_dict)
    """)
