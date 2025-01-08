"""
This module handles all data processing tasks for the Titanic dataset analysis.
It includes functions for cleaning, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Union, List, Tuple, Optional, Dict
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class DataLoader:
    """Class to handle data loading operations."""
    
    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        return pd.read_csv(file_path)
    
    @classmethod
    def load_multiple_csv(cls, file_paths: List[Union[str, Path]]) -> List[pd.DataFrame]:
        """
        Load multiple CSV files into DataFrames.
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            List of DataFrames containing the loaded data
        """
        return [cls.load_csv(path) for path in file_paths]

class FeatureProcessor:
    """Class to handle feature processing operations."""
    
    def __init__(self):
        self.scaler = None
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset using intelligent imputation.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Age: Use random values within mean ± std based on Pclass and Sex
        age_mean = df.groupby(['Pclass', 'Sex'])['Age'].transform('mean')
        age_std = df.groupby(['Pclass', 'Sex'])['Age'].transform('std')
        age_null = df['Age'].isnull()
        # Fix: Generate random ages for each missing value individually
        for idx in df[age_null].index:
            # Get values and handle potential NaN
            current_mean = age_mean.iloc[idx]
            current_std = age_std.iloc[idx]

            # Use fallback values if needed
            if np.isnan(current_mean):
                current_mean = df['Age'].mean()
            if np.isnan(current_std) or current_std == 0:
                current_std = df['Age'].std() if not np.isnan(df['Age'].std()) else 1.0

            # Generate random age
            random_age = np.random.normal(current_mean, current_std)
            df.loc[idx, 'Age'] = random_age
        # Ensure age is not negative
        df.loc[df['Age'] < 0, 'Age'] = 0
        
        # Fare: Use median based on Pclass
        df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
        
        # Embarked: Use most frequent value
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        # Cabin: Create new feature indicating if cabin is known
        df['HasCabin'] = (~df['Cabin'].isnull()).astype(int)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            df: Input dataframe
            categorical_features: List of categorical features to encode
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        if categorical_features is None:
            # Include both object and category dtypes
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Add AgeGroup if it exists (it's created in create_features)
            if 'AgeGroup' in df.columns and 'AgeGroup' not in categorical_features:
                categorical_features.append('AgeGroup')
        
        # Initialize an empty DataFrame for the encoded features
        encoded_df = df.select_dtypes(exclude=['object', 'category']).copy()
        
        # One-hot encode each categorical feature
        for feature in categorical_features:
            if feature in df.columns:
                # Create dummy variables and drop the first category to avoid multicollinearity
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
                # Add the dummy columns to the dataset
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
        
        return encoded_df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        
        # Family size and is_alone
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Fare per person
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        
        # Age categories
        df['AgeGroup'] = pd.cut(df['Age'], 
                              bins=[0, 12, 18, 35, 50, 65, 100],
                              labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior', 'Elderly'])
        
        # Title from Name
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }
        df['Title'] = df['Title'].map(title_mapping)
        
        # Cabin prefix (deck)
        df['Deck'] = df['Cabin'].str.slice(0, 1)
        df['Deck'] = df['Deck'].fillna('Unknown')
        
        return df
    
    def scale_features(self, 
                      df: pd.DataFrame, 
                      features_to_scale: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input dataframe
            features_to_scale: List of features to scale
            
        Returns:
            Scaled DataFrame
        """
        df = df.copy()
        
        if features_to_scale is None:
            # Scale all numeric columns
            features_to_scale = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Only scale features that exist in the dataframe
        features_to_scale = [f for f in features_to_scale if f in df.columns]
        
        if features_to_scale:  # Only scale if there are features to scale
            if self.scaler is None:
                self.scaler = StandardScaler()
                df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
            else:
                df[features_to_scale] = self.scaler.transform(df[features_to_scale])
        
        return df

class ModelEvaluator:
    """Class to handle model training and evaluation."""
    
    def __init__(self, visualizer):
        """
        Initialize ModelEvaluator.
        
        Args:
            visualizer: TitanicVisualizer instance for plotting
        """
        self.visualizer = visualizer
        self.default_models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=0.1
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=10
            ),
            'SVM': SVC(
                probability=True,
                random_state=42,
                C=1.0,
                kernel='rbf',
                class_weight='balanced'
            )
        }
        self.default_param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def train_and_evaluate_model(self, model, X_train, X_val, y_train, y_val):
        """
        Train and evaluate a model with comprehensive metrics.
        
        Args:
            model: The model to train and evaluate
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Dictionary containing various performance metrics
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
            # Plot ROC curve
            self.visualizer.plot_roc_curve(y_val, y_pred_proba, model.__class__.__name__)
        
        # Plot confusion matrix
        self.visualizer.plot_confusion_matrix(y_val, y_pred)
        
        return metrics
    
    def evaluate_all_models(self, X_train, X_val, y_train, y_val, models=None):
        """
        Train and evaluate multiple models.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            models: Dictionary of models to evaluate. If None, uses default models.
            
        Returns:
            DataFrame containing performance metrics for all models
        """
        if models is None:
            models = self.default_models
            
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            results[name] = self.train_and_evaluate_model(model, X_train, X_val, y_train, y_val)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results).round(3)
        
        # Plot model comparison
        self.visualizer.plot_model_comparison(results_df)
        
        return results_df
    
    def perform_grid_search(self, X_train, X_val, y_train, y_val, models=None, param_grids=None):
        """
        Perform grid search to find optimal parameters for each model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            models: Dictionary of models to optimize. If None, uses default models.
            param_grids: Dictionary of parameter grids. If None, uses default grids.
            
        Returns:
            Tuple of (DataFrame with results, Dictionary of optimized models)
        """
        if models is None:
            models = self.default_models
        if param_grids is None:
            param_grids = self.default_param_grids
            
        optimized_models = {}
        optimized_results = {}
        
        for name, model in models.items():
            print(f"\nOptimizing {name}...")
            grid_search = GridSearchCV(
                model,
                param_grids[name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            optimized_models[name] = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            optimized_results[name] = self.train_and_evaluate_model(
                grid_search.best_estimator_,
                X_train, X_val, y_train, y_val
            )
        
        # Create results DataFrame
        results_df = pd.DataFrame(optimized_results).round(3)
        
        # Plot model comparison
        self.visualizer.plot_model_comparison(results_df)
        
        return results_df, optimized_models
    
    def get_best_model_and_predict(self, results_df, optimized_models, test_data):
        """
        Get the best model and make predictions on test data.
        
        Args:
            results_df: DataFrame containing model performance metrics
            optimized_models: Dictionary of trained models
            test_data: Test data to make predictions on
            
        Returns:
            Tuple of (best model name, submission DataFrame)
        """
        # Select best model based on validation accuracy
        # The DataFrame has models as columns and metrics as rows
        accuracies = results_df.loc['accuracy']
        best_model_name = accuracies.idxmax()
        best_model = optimized_models[best_model_name]
        
        print(f"Best model: {best_model_name}")
        
        # Generate predictions
        final_predictions = best_model.predict(test_data)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': test_data['PassengerId'],
            'Survived': final_predictions
        })
        
        return best_model_name, submission
    
    def save_submission(self, submission, filename='submission.csv'):
        """
        Save submission DataFrame to file.
        
        Args:
            submission: DataFrame containing predictions
            filename: Name of the file to save
            
        Returns:
            Path to saved file
        """
        # Create submissions directory if it doesn't exist
        submission_dir = Path('../submissions')
        submission_dir.mkdir(exist_ok=True)
        
        # Save predictions
        submission_path = submission_dir / filename
        submission.to_csv(submission_path, index=False)
        
        print(f"Submission file saved to: {submission_path}")
        print("\nSample predictions:")
        print(submission.head())
        
        return submission_path

class TitanicPreprocessor:
    """Main class for Titanic dataset preprocessing."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_processor = FeatureProcessor()
        
    def prepare_data(self, 
                    train_path: Union[str, Path],
                    test_path: Union[str, Path],
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Prepare data for modeling by applying all preprocessing steps.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            test_size: Validation set size
            random_state: Random seed
            
        Returns:
            Dictionary containing:
                - X_train: Training features
                - X_val: Validation features
                - y_train: Training labels
                - y_val: Validation labels
                - test_processed: Processed test set
        """
        # Load data
        train_df = self.data_loader.load_csv(train_path)
        test_df = self.data_loader.load_csv(test_path)
        
        # Store PassengerId for test set
        test_ids = test_df['PassengerId']
        
        # Extract target and encode it
        y = train_df['Survived']
        train_df = train_df.drop('Survived', axis=1)
        
        # Combine train and test for preprocessing
        all_data = pd.concat([train_df, test_df], sort=False)
        
        # Apply preprocessing steps
        all_data = self.feature_processor.handle_missing_values(all_data)
        all_data = self.feature_processor.create_features(all_data)
        all_data = self.feature_processor.encode_categorical_features(all_data)
        all_data = self.feature_processor.scale_features(all_data)
        
        # Split back into train and test
        train_processed = all_data[:len(train_df)]
        test_processed = all_data[len(train_df):]
        
        # Create train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            train_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Add PassengerId back to test set
        test_processed.loc[:, 'PassengerId'] = test_ids
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'test_processed': test_processed
        }
