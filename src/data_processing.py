"""
This module handles all data processing tasks for the Titanic dataset analysis.
It includes functions for cleaning, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """
    Load training and test datasets.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_clean = df.copy()
    
    # Age: Fill with median
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    
    # Embarked: Fill with most common value
    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Fare: Fill with median
    if 'Fare' in df_clean.columns:
        df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
    
    return df_clean

def encode_categorical_features(df):
    """
    Encode categorical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
    """
    df_encoded = df.copy()
    
    # Sex encoding
    df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
    
    # Embarked encoding
    if 'Embarked' in df_encoded.columns:
        embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
        df_encoded['Embarked'] = df_encoded['Embarked'].map(embarked_mapping)
    
    return df_encoded

def create_features(df):
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    df_featured = df.copy()
    
    # Family size
    df_featured['FamilySize'] = df_featured['SibSp'] + df_featured['Parch'] + 1
    
    # Is alone
    df_featured['IsAlone'] = (df_featured['FamilySize'] == 1).astype(int)
    
    # Fare per person
    df_featured['FarePerPerson'] = df_featured['Fare'] / df_featured['FamilySize']
    
    # Age groups
    df_featured['AgeGroup'] = pd.cut(df_featured['Age'], 
                                   bins=[0, 12, 18, 35, 50, 100],
                                   labels=[0, 1, 2, 3, 4])
    
    return df_featured

def scale_features(df, scaler=None, features_to_scale=None):
    """
    Scale numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        scaler (StandardScaler, optional): Fitted scaler for transformation
        features_to_scale (list, optional): List of features to scale
        
    Returns:
        tuple: (scaled_df, scaler)
    """
    df_scaled = df.copy()
    
    if features_to_scale is None:
        features_to_scale = ['Age', 'Fare', 'FamilySize', 'FarePerPerson']
    
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    else:
        df_scaled[features_to_scale] = scaler.transform(df_scaled[features_to_scale])
    
    return df_scaled, scaler

def prepare_data(train_df, test_df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling by applying all preprocessing steps.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        test_size (float): Validation set size
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, test_processed, scaler)
    """
    # Handle missing values
    train_clean = handle_missing_values(train_df)
    test_clean = handle_missing_values(test_df)
    
    # Encode categorical features
    train_encoded = encode_categorical_features(train_clean)
    test_encoded = encode_categorical_features(test_clean)
    
    # Create new features
    train_featured = create_features(train_encoded)
    test_featured = create_features(test_encoded)
    
    # Prepare features and target
    features_to_drop = ['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId']
    X = train_featured.drop([col for col in features_to_drop if col in train_featured.columns], axis=1)
    y = train_featured['Survived'] if 'Survived' in train_featured.columns else None
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    test_scaled, _ = scale_features(test_featured.drop([col for col in features_to_drop if col in test_featured.columns], axis=1), 
                                  scaler=scaler)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, 
                                                      test_size=test_size, 
                                                      random_state=random_state)
    
    return X_train, X_val, y_train, y_val, test_scaled, scaler
