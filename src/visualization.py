"""
This module provides visualization functions for the Titanic dataset analysis.
It includes various plots for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def set_plotting_style():
    """Set the default plotting style."""
    plt.style.use('seaborn')
    sns.set_palette("husl")

def plot_survival_by_feature(df, feature, title=None):
    """
    Plot survival rate by a specific feature.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature to analyze
        title (str, optional): Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature, y='Survived', data=df)
    plt.title(title or f'Survival Rate by {feature}')
    plt.ylabel('Survival Rate')
    plt.tight_layout()

def plot_age_distribution(df):
    """
    Plot age distribution by survival status.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple="stack", bins=30)
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()

def plot_feature_importance(feature_importance_df):
    """
    Plot feature importance from a model.
    
    Args:
        feature_importance_df (pd.DataFrame): DataFrame with feature names and importance scores
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance Analysis')
    plt.xlabel('Importance Score')
    plt.tight_layout()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list, optional): Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels)
        plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    plt.tight_layout()

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        model_name (str): Name of the model
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

def plot_model_comparison(models_df):
    """
    Plot model comparison across different metrics.
    
    Args:
        models_df (pd.DataFrame): DataFrame with model names and their metrics
    """
    plt.figure(figsize=(12, 6))
    models_df.plot(kind='bar', width=0.8)
    plt.title('Model Comparison Across Metrics')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def create_analysis_plots(df):
    """
    Create a comprehensive set of analysis plots.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    set_plotting_style()
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Survival by Gender
    plt.subplot(2, 2, 1)
    plot_survival_by_feature(df, 'Sex', 'Survival Rate by Gender')
    
    # 2. Survival by Passenger Class
    plt.subplot(2, 2, 2)
    plot_survival_by_feature(df, 'Pclass', 'Survival Rate by Passenger Class')
    
    # 3. Age Distribution
    plt.subplot(2, 2, 3)
    plot_age_distribution(df)
    
    # 4. Correlation Matrix
    plt.subplot(2, 2, 4)
    plot_correlation_matrix(df)
    
    plt.tight_layout()
    plt.show()
