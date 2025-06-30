#preping data for ml model(s)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import os

from google.colab import drive 
drive.mount('/content/drive')


def preprocess_real_estate_data(df, target_column='price'):

	# Droping irrelevant columns
    columns_to_drop = ['full_address', 'street', 'sold_date', 'status']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped columns: {columns_to_drop}")
    else:
        print("Not found.")

    #numerical and categorical features
    feature_columns = [col for col in df.columns if col != target_column] if target_column else df.columns.tolist()

    numerical_features = []
    categorical_features = []

    #loop though to assign as num or cat
    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            if col in ['bed', 'bath', 'acre_lot', 'house_size']: # true numericals
                 numerical_features.append(col)
            elif col == 'zip_code': # zip_code is categorical
                 df[col] = df[col].astype(str) #converting to string
                 categorical_features.append(col)
            elif col != target_column : # other numerical
                 numerical_features.append(col)
        else: # categorical
            categorical_features.append(col)
            df[col] = df[col].astype(str) # ensure string type for OHE

    # Separating Target Variable (if specified)
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df[feature_columns].copy() #(SettingWithCopyWarning)
        print(f"'{target_column}' separated.")
    elif target_column:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    else: # no target specified
        X = df.copy()
        print("No target column specified, processing entire DataFrame as features.")

    # Create Preprocessing Pipelines
    # Numerical pipeline. median imputation and scaling
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline.Imputw with 'missing' and OneHotEncode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Dense array output
    ])

    # Apply transformers to columns
    final_numerical_features = [f for f in numerical_features if f in X.columns]
    final_categorical_features = [f for f in categorical_features if f in X.columns]

    transformers_list = []
    if final_numerical_features:
        transformers_list.append(('num', numerical_pipeline, final_numerical_features))
    if final_categorical_features:
        transformers_list.append(('cat', categorical_pipeline, final_categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')

    #Apply prep
    X_processed_np = preprocessor.fit_transform(X)

    # Convert processed data back to df
    # reconstruct DataFrame with new feature names from ohe.
    processed_feature_names = []
    if 'num' in preprocessor.named_transformers_ and final_numerical_features:
        processed_feature_names.extend(final_numerical_features) # Numerical names remain
    
    if 'cat' in preprocessor.named_transformers_ and final_categorical_features:
        try:
            onehot_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(final_categorical_features)
            processed_feature_names.extend(onehot_cols)
        except KeyError: 
             print("Categorical transformer 'cat' not found during feature name reconstruction.")


    X_processed = pd.DataFrame(X_processed_np, columns=processed_feature_names, index=X.index)

    print(f"Preprocessing complete. Processed DataFrame shape: {X_processed.shape}")

    if target_column:
        return X_processed, y
    else:
        return X_processed

if __name__ == '__main__':

    data_file_path = "/content/drive/My Drive/ML_project/ML_Project/Random Forest Regressor module - Final/2 realtor-data.csv"
    # Output directory
    output_dir = os.path.dirname(data_file_path)
    

    #Load data
    try:
        df_loaded = pd.read_csv(data_file_path) 
        print(df_loaded.head())
        df_loaded.info()
        
        #Predicting 'price'
        df_for_price_pred = df_loaded.copy() # Using a fresh copy
        X_processed_price, y_price = preprocess_real_estate_data(df_for_price_pred, target_column='price')
        
        print(X_processed_price.head())
        print(y_price.head())

        # Combine processed features and target for saving
        processed_df_price = X_processed_price.copy()
        processed_df_price['price'] = y_price # Add target column back
        
        output_price_path = os.path.join(output_dir, "prep-realtor-data_price.csv")
        processed_df_price.to_csv(output_price_path, index=False)
        print(f"\nPreprocessed data for price prediction saved to: {output_price_path}")



    except FileNotFoundError:
        print(f"Error: The file '{data_file_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")