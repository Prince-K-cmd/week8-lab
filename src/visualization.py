"""
This module provides visualization functions for the Titanic dataset analysis.
It includes various plots for data exploration and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Optional, List, Union, Dict, Any
import os

class PlotConfig:
    """Configuration for plot styling and paths."""
    
    FIGURE_SIZE = (12, 8)
    STYLE = 'seaborn-v0_8'
    PALETTE = 'husl'
    DPI = 100
    
    # Plot directories
    BASE_DIR = '../plots'
    EXPLORATORY_DIR = os.path.join(BASE_DIR, 'exploratory')
    MODEL_EVALUATION_DIR = os.path.join(BASE_DIR, 'model_evaluation')
    FEATURE_IMPORTANCE_DIR = os.path.join(BASE_DIR, 'feature_importance')
    
    @classmethod
    def set_style(cls) -> None: 
        """Set the default plotting style."""
        plt.style.use(cls.STYLE)
        sns.set_palette(cls.PALETTE)
        plt.rcParams['figure.dpi'] = cls.DPI
    
    @classmethod
    def create_plot_dirs(cls) -> None:
        """Create plot directories if they don't exist."""
        for directory in [cls.BASE_DIR, cls.EXPLORATORY_DIR, 
                         cls.MODEL_EVALUATION_DIR, cls.FEATURE_IMPORTANCE_DIR]:
            os.makedirs(directory, exist_ok=True)

class TitanicVisualizer:
    """Class for creating visualizations for Titanic dataset analysis."""
    
    def __init__(self):
        PlotConfig.set_style()
    
    def plot_survival_by_feature(self, 
                               df: pd.DataFrame, 
                               feature: str,
                               title: Optional[str] = None,
                               figsize: Optional[tuple] = None,
                               save: bool = True) -> None:
        """
        Plot survival rate by a specific feature.
        
        Args:
            df: Input dataframe
            feature: Feature to analyze
            title: Plot title
            figsize: Figure size tuple (width, height)
            save: Whether to save the plot
        """
        plt.figure(figsize=figsize or PlotConfig.FIGURE_SIZE)
        
        if df[feature].dtype in ['int64', 'float64']:
            # For numerical features, use a line plot
            survival_rates = df.groupby(feature)['Survived'].mean()
            plt.plot(survival_rates.index, survival_rates.values, marker='o')
        else:
            # For categorical features, use a bar plot
            sns.barplot(x=feature, y='Survived', data=df, errorbar=('ci', 95))
        
        plt.title(title or f'Survival Rate by {feature}')
        plt.ylabel('Survival Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            # Create exploratory directory if it doesn't exist
            os.makedirs(PlotConfig.EXPLORATORY_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.EXPLORATORY_DIR, f'survival_by_{feature}.png'))
            plt.close()
    
    def plot_age_distribution(self, 
                            df: pd.DataFrame,
                            hue: Optional[str] = 'Survived',
                            figsize: Optional[tuple] = None,
                            save: bool = True) -> None:
        """
        Plot age distribution with optional grouping.
        
        Args:
            df: Input dataframe
            hue: Column to use for grouping (default: 'Survived')
            figsize: Figure size tuple (width, height)
            save: Whether to save the plot
        """
        plt.figure(figsize=figsize or PlotConfig.FIGURE_SIZE)
        sns.kdeplot(data=df, x='Age', hue=hue, common_norm=False)
        plt.title('Age Distribution by ' + hue)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            # Create exploratory directory if it doesn't exist
            os.makedirs(PlotConfig.EXPLORATORY_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.EXPLORATORY_DIR, 'age_distribution.png'))
            plt.close()
    
    def plot_correlation_matrix(self, 
                              df: pd.DataFrame,
                              figsize: Optional[tuple] = None,
                              save: bool = True) -> None:
        """
        Plot correlation matrix of numerical features.
        
        Args:
            df: Input dataframe
            figsize: Figure size tuple (width, height)
            save: Whether to save the plot
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=figsize or PlotConfig.FIGURE_SIZE)
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(numeric_df.corr(), 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save:
            # Create exploratory directory if it doesn't exist
            os.makedirs(PlotConfig.EXPLORATORY_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.EXPLORATORY_DIR, 'correlation_matrix.png'))
            plt.close()
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            save: bool = True) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=PlotConfig.FIGURE_SIZE)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save:
            # Create model evaluation directory if it doesn't exist
            os.makedirs(PlotConfig.MODEL_EVALUATION_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.MODEL_EVALUATION_DIR, 'confusion_matrix.png'))
            plt.close()
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      model_name: str,
                      save: bool = True) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=PlotConfig.FIGURE_SIZE)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            # Create model evaluation directory if it doesn't exist
            os.makedirs(PlotConfig.MODEL_EVALUATION_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.MODEL_EVALUATION_DIR, f'roc_curve_{model_name}.png'))
            plt.close()
    
    def plot_feature_importance(self, 
                              feature_importance_df: pd.DataFrame,
                              save: bool = True) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance_df: DataFrame with feature importance scores
            save: Whether to save the plot
        """
        plt.figure(figsize=PlotConfig.FIGURE_SIZE)
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save:
            # Create feature importance directory if it doesn't exist
            os.makedirs(PlotConfig.FEATURE_IMPORTANCE_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.FEATURE_IMPORTANCE_DIR, 'feature_importance.png'))
            plt.close()
    
    def plot_model_comparison(self, 
                            results_df: pd.DataFrame,
                            save: bool = True) -> None:
        """
        Plot model comparison metrics.
        
        Args:
            results_df: DataFrame with model performance metrics
            save: Whether to save the plot
        """
        plt.figure(figsize=PlotConfig.FIGURE_SIZE)
        results_df.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save:
            # Create model evaluation directory if it doesn't exist
            os.makedirs(PlotConfig.MODEL_EVALUATION_DIR, exist_ok=True)
            plt.savefig(os.path.join(PlotConfig.MODEL_EVALUATION_DIR, 'model_comparison.png'))
            plt.close()
