# Titanic Data Analysis Project

## Overview

This project contains a detailed analysis of the Titanic dataset, exploring various aspects of the passengers and factors that influenced survival rates.

## Project Structure

```
titanic_wi/
│
├── data/               # Data directory
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
│
├── notebooks/         # Jupyter notebooks for analysis
│   └── analysis.ipynb
│
├── src/              # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   └── visualization.py
│
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setting up the Development Environment

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd titanic_wi
   ```

2. Create a virtual environment:

   ```bash
   # Windows
   python -m venv .venv

   # Linux/MacOS
   python3 -m venv .venv
   ```

3. Activate the virtual environment:

   ```bash
   # Windows (Command Prompt)
   .venv\Scripts\activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # Linux/MacOS
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Deactivate the virtual environment when you're done:

   ```bash
   deactivate
   ```

### Development Best Practices

- Always activate the virtual environment before working on the project
- Update requirements.txt if you add new dependencies:

  ```bash
  pip freeze > requirements.txt
  ```

- Keep the virtual environment isolated from other projects

## Analysis Questions

The analysis covers various aspects including:

- Passenger demographics
- Survival rates analysis
- Feature importance
- Statistical analysis
- Visualization of key findings

## Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
