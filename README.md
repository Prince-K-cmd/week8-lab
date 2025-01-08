# Titanic Survival Prediction Project

## Overview

This project implements a machine learning solution to predict passenger survival on the Titanic using various classification algorithms. The implementation features a modular, object-oriented design with comprehensive data preprocessing, model evaluation, and visualization capabilities.

## Project Structure

```python
Titanic_wi/
├── data/               # Data directory
│   ├── train.csv      # Training dataset
│   └── test.csv       # Test dataset
├── notebooks/         # Jupyter notebooks
│   └── titanic_analysis.ipynb
├── plots/             # Generated plots
│   ├── exploratory/
│   ├── model_evaluation/
│   └── feature_importance/
├── src/              # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   └── visualization.py
├── submissions/      # Model predictions
├── requirements.txt  # Project dependencies
└── report.md        # Detailed project report
```

## Features

- Modular code structure with separate modules for data processing and visualization
- Comprehensive feature engineering and preprocessing pipeline
- Multiple machine learning models (Logistic Regression, Random Forest, SVM)
- Grid search for hyperparameter optimization
- Extensive visualization capabilities
- Cross-validation and model evaluation metrics

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd Titanic_wi
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:

```bash
jupyter notebook notebooks/titanic_analysis.ipynb
```

2. Follow the notebook for:
   - Data preprocessing and feature engineering
   - Model training and optimization
   - Performance evaluation
   - Generating predictions

## Code Structure

- `DataLoader`: Handles data loading operations
- `FeatureProcessor`: Manages feature engineering and preprocessing
- `TitanicPreprocessor`: Main preprocessing pipeline
- `ModelEvaluator`: Handles model training, evaluation, and optimization
- `TitanicVisualizer`: Creates various visualizations

## Results

For detailed analysis and results, please refer to [report.md](report.md).

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter notebook
