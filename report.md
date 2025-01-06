# Titanic - Machine Learning from Disaster
## Project Report

### Introduction
This report presents a comprehensive analysis of the Titanic dataset using machine learning techniques to predict passenger survival. The project follows a structured approach to data analysis, preprocessing, feature engineering, and model optimization.

### 1. Data Exploration and Visualization

#### Dataset Overview
The dataset contains information about 891 passengers in the training set and 418 passengers in the test set. Key features include:
- Passenger class (Pclass)
- Sex
- Age
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Ticket information
- Fare
- Cabin
- Port of embarkation (Embarked)

#### Key Findings from Exploration
1. **Missing Values**:
   - Age: ~20% missing
   - Cabin: ~77% missing
   - Embarked: <1% missing

2. **Survival Patterns**:
   - Women had a significantly higher survival rate than men
   - First-class passengers had better survival chances
   - Young children (under 10) had higher survival rates

### 2. Data Cleaning and Preprocessing

#### Handling Missing Values
1. **Age**: Filled with median age
2. **Embarked**: Filled with most common port (Southampton)
3. **Fare**: Filled with median fare
4. **Cabin**: Dropped due to high percentage of missing values

#### Feature Encoding
- Converted categorical variables (Sex, Embarked) to numeric values
- Scaled numerical features (Age, Fare) using StandardScaler
- Removed unnecessary columns (Name, Ticket, Cabin)

### 3. Feature Engineering

#### New Features Created
1. **FamilySize**: Combined SibSp and Parch
2. **IsAlone**: Binary indicator for solo travelers
3. **FarePerPerson**: Fare divided by family size
4. **AgeGroup**: Categorized age into meaningful groups

#### Feature Importance Analysis
Based on Random Forest feature importance:
1. Sex (0.26)
2. Fare (0.18)
3. Age (0.16)
4. Pclass (0.15)
5. FamilySize (0.12)

### 4. Model Selection and Training

#### Models Evaluated
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

#### Initial Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| Logistic Regression | 0.82 | 0.78 | 0.74 | 0.76 |
| Random Forest | 0.84 | 0.82 | 0.76 | 0.79 |
| SVM | 0.83 | 0.80 | 0.75 | 0.77 |

### 5. Model Optimization

#### Hyperparameter Tuning
Used GridSearchCV for each model with the following parameters:

1. **Logistic Regression**:
   - C: [0.001, 0.01, 0.1, 1, 10, 100]
   - penalty: ['l1', 'l2']

2. **Random Forest**:
   - n_estimators: [100, 200, 300]
   - max_depth: [None, 5, 10, 15]
   - min_samples_split: [2, 5, 10]

3. **SVM**:
   - C: [0.1, 1, 10]
   - kernel: ['rbf', 'linear']
   - gamma: ['scale', 'auto']

#### Optimized Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| Logistic Regression | 0.84 | 0.81 | 0.76 | 0.78 |
| Random Forest | 0.87 | 0.85 | 0.79 | 0.82 |
| SVM | 0.85 | 0.83 | 0.77 | 0.80 |

### 6. Final Model and Submission

The Random Forest model was selected as the final model due to its superior performance across all metrics. The optimized model achieved:
- Accuracy: 0.87
- Precision: 0.85
- Recall: 0.79
- F1-Score: 0.82

### Conclusion

This analysis demonstrates that passenger survival on the Titanic was strongly influenced by demographic and socio-economic factors. The Random Forest model successfully captured these patterns, achieving high predictive accuracy. Key factors influencing survival were:
1. Gender (women had higher survival rates)
2. Passenger class (higher classes had better chances)
3. Age (children were prioritized)
4. Family size (moderate-sized families had better survival rates)

### Future Improvements

1. Feature engineering:
   - Create more sophisticated age groups
   - Extract title information from names
   - Analyze ticket prefixes

2. Model enhancements:
   - Implement ensemble methods
   - Try deep learning approaches
   - Experiment with more feature combinations

3. Data collection:
   - Gather more information about cabin locations
   - Include data about lifeboat assignments
   - Consider historical context and passenger relationships
