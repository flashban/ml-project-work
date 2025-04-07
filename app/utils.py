import pandas as pd
import numpy as np
import streamlit as st

def load_dataset(file_path):
    """
    Load the CO2 emissions dataset with error handling
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset or None if error occurs
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def clean_dataset(df):
    """
    Clean the dataset by handling missing values
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove rows with missing values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned

def get_categorical_columns(df):
    """
    Get categorical columns from the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        list: List of categorical column names
    """
    return df.select_dtypes(include=['object']).columns.tolist()

def get_numeric_columns(df):
    """
    Get numeric columns from the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        list: List of numeric column names
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a specific column
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
    
    Returns:
        dict: Summary statistics
    """
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'min': df[column].min(),
        'max': df[column].max(),
        'std': df[column].std()
    }

def detect_outliers(df, column, method='iqr'):
    """
    Detect outliers in a dataset column
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to check for outliers
        method (str): Method to detect outliers ('iqr' or 'zscore')
    
    Returns:
        pd.DataFrame: Rows with outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3]
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return outliers

def prepare_input_data(input_dict):
    """
    Prepare input data for model prediction
    
    Args:
        input_dict (dict): Dictionary of input features
    
    Returns:
        pd.DataFrame: Prepared input dataframe
    """
    input_df = pd.DataFrame([input_dict])
    return input_df

def validate_input_data(input_df, valid_categories):
    """
    Validate input data against known categories
    
    Args:
        input_df (pd.DataFrame): Input dataframe
        valid_categories (dict): Dictionary of valid categories for each feature
    
    Returns:
        bool: Whether input is valid
    """
    for column, categories in valid_categories.items():
        if column in input_df.columns:
            if input_df[column].iloc[0] not in categories:
                return False
    return True

def create_feature_description():
    """
    Create a description of features used in the model
    
    Returns:
        dict: Feature descriptions
    """
    return {
        'Make': 'Vehicle manufacturer',
        'Model': 'Specific vehicle model',
        'Vehicle Class': 'Category of vehicle',
        'Engine Size(L)': 'Volume of the engine in liters',
        'Cylinders': 'Number of cylinders in the engine',
        'Transmission': 'Type of transmission',
        'Fuel Type': 'Type of fuel used by the vehicle',
        'CO2 Emissions(g/km)': 'Carbon dioxide emissions per kilometer'
    }

def log_prediction(input_data, prediction):
    """
    Log prediction details (can be expanded to write to a file or database)
    
    Args:
        input_data (pd.DataFrame): Input features
        prediction (float): Predicted CO2 emissions
    """
    log_entry = {
        'timestamp': pd.Timestamp.now(),
        'input_data': input_data.to_dict(),
        'predicted_emissions': prediction
    }
    # In a real-world scenario, you might want to write this to a log file or database
    print("Prediction Logged:", log_entry)

def get_data_sample(df, n=5):
    """
    Get a sample of the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        n (int): Number of samples to return
    
    Returns:
        pd.DataFrame: Sample of the dataset
    """
    return df.sample(n)