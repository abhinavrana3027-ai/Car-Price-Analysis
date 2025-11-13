#!/usr/bin/env python3
"""Main execution script for Car Price Analysis project.

This script runs the complete ML pipeline end-to-end.
Simply run: python run_analysis.py

Author: Abhinav Rana
Date: November 2025
"""

import sys
import os

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from car_price_analysis import CarPricePredictor

def main():
    """
    Main execution function that runs the complete analysis pipeline.
    """
    print("="*80)
    print("CAR PRICE PREDICTION - MACHINE LEARNING PIPELINE")
    print("Author: Abhinav Rana")
    print("="*80)
    print()
    
    print("🚀 Starting analysis pipeline...")
    print()
    
    try:
        # Initialize the pipeline
        pipeline = CarPricePredictor('car_price_prediction_.csv')
        
        # Execute complete pipeline
        (
            pipeline
            .load_data()
            .perform_eda()
            .feature_engineering()
            .preprocess_data()
            .train_models()
            .visualize_results()
            .print_summary()
        )
        
        print()
        print("✅ Analysis pipeline completed successfully!")
        print()
        print("Generated files:")
        print("  - eda_price_distribution.png")
        print("  - correlation_heatmap.png")
        print("  - model_comparison.png")
        print("  - actual_vs_predicted.png")
        print()
        print("Check the plots to see visualizations!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Data file not found - {str(e)}")
        print("\nPlease ensure 'car_price_prediction_.csv' is in the same directory.")
        sys.exit(1)
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies - {str(e)}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("\nPlease check your data file and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
