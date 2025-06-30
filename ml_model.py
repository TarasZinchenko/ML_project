# This script loads prep data, trains a machine learning model and evaluates its performance. +visuals

import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns # Added for the feature importance plot
from sklearn.tree import plot_tree

# Target variable
TARGET_VARIABLE_NAME = 'price' 

# Path to preped data
PREPROCESSED_DATA_PATH = "/content/drive/My Drive/ML_project/ML_Project/Random Forest Regressor module - Final/prep-realtor-data_price.csv"

'''Main script, ml training'''
#Try to load 
try:
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
except FileNotFoundError:
    print(f"Not found at '{PREPROCESSED_DATA_PATH}'. Run the data_preprocessor.py")
except Exception as e:
    print(f"Error: {e}")

# Proceed only if the DataFrame was loaded successfully
if 'df' in locals():
    print(df.head())
    df.info()

    # Check if target exists
    if TARGET_VARIABLE_NAME not in df.columns:
        print(f"Target '{TARGET_VARIABLE_NAME}' not founw.")
    else:
        # separate X and Y
        X = df.drop(columns=[TARGET_VARIABLE_NAME])
        y = df[TARGET_VARIABLE_NAME]

        print("\n")
        print(X.shape)
        print(y.shape)

        #Initialize model and task type
        # Using n_jobs=-1 will use all available CPU cores and significantly speed up training
        model = RandomForestRegressor(random_state=42, n_jobs=-1) 
        #to do: more regression models(if needed)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"X_test {X_test.shape}, y_test {y_test.shape}")

        # Train the model
        print("\nStarting model training (this may take a while)...")
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate the model
        y_pred = model.predict(X_test)

        # print RMSE and R2 score
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2 Score): {r2:.4f}")

        '''Below is everything for visualization'''

        
        # actual over predicted prices
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.3)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, 'r-', linewidth=2)
        plt.xlabel("Actual prices")
        plt.ylabel("Predicted prices")
        plt.title("Actual over Predicted prices")
        plt.xlim(lims)
        plt.ylim(lims)
        plt.grid(True)
        plt.savefig('actual_over_predicted.png')
        plt.show() #For colab or Jupiter Notebook

        # Residuals plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Prices")
        plt.ylabel("Residuals (Error)")
        plt.title("Residuals vs. Predicted Prices")
        plt.grid(True)
        plt.savefig('residuals_plot.png')
        plt.show() 

        # Importance of the features
        importances = model.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
        plt.title('Top 20 most important features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()