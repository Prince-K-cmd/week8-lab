"""
This module contains functions for processing and cleaning the Titanic dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load the Titanic dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Clean the Titanic dataset by handling missing values and converting datatypes.
    
    Args:
        df (pd.DataFrame): Raw Titanic dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    
    # Convert categorical variables
    df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
    df_clean['Embarked'] = df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df_clean

def feature_engineering(df):
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    df_featured = df.copy()
    
    # Extract title from name
    df_featured['Title'] = df_featured['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Create family size feature
    df_featured['FamilySize'] = df_featured['SibSp'] + df_featured['Parch'] + 1
    
    # Create is_alone feature
    df_featured['IsAlone'] = (df_featured['FamilySize'] == 1).astype(int)
    
    return df_featured

def prepare_features(df, scaler=None):
    """
    Prepare features for modeling by selecting and scaling relevant features.
    
    Args:
        df (pd.DataFrame): Dataset with engineered features
        scaler (StandardScaler, optional): Fitted scaler for transformation
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    # Select features for modeling
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']
    X = df[features].copy()
    y = df['Survived'] if 'Survived' in df.columns else None
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    return X, y, scaler
