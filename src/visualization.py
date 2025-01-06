"""
This module contains functions for visualizing the Titanic dataset analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_plotting_style():
    """Set the style for all visualizations."""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_survival_by_feature(df, feature, title=None):
    """
    Create a bar plot showing survival rates by a specific feature.
    
    Args:
        df (pd.DataFrame): The dataset
        feature (str): The feature to analyze
        title (str, optional): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature, y='Survived', data=df)
    plt.title(title or f'Survival Rate by {feature}')
    plt.ylabel('Survival Rate')
    plt.xlabel(feature)
    plt.tight_layout()

def plot_age_distribution(df):
    """
    Plot the age distribution of passengers.
    
    Args:
        df (pd.DataFrame): The dataset
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple="stack", bins=30)
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()

def plot_correlation_matrix(df):
    """
    Plot a correlation matrix of numerical features.
    
    Args:
        df (pd.DataFrame): The dataset
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()

def plot_feature_importance(feature_importance, feature_names):
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_importance (array): Array of feature importance scores
        feature_names (list): List of feature names
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()

def plot_survival_by_class_and_sex(df):
    """
    Create a grouped bar plot showing survival rates by passenger class and sex.
    
    Args:
        df (pd.DataFrame): The dataset
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
    plt.title('Survival Rate by Passenger Class and Sex')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')
    plt.tight_layout()
