# 🏦 Credit Risk Scoring Model



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning solution to predict loan default risk using Home Credit data. This project helps financial institutions identify high-risk applicants to minimize non-performing loans (NPLs).

## 📌 Key Features

- **Data Processing Pipeline**: Handles missing values, outliers, and merges 7 relational tables
- **Feature Engineering**: Creates 50+ predictive features including payment ratios and credit history aggregates
- **Model Comparison**: Evaluates LightGBM, Random Forest, and Logistic Regression
- **Best Model**: LightGBM with **AUC-ROC 0.784**
- **Production-Ready**: Modular code structure for easy deployment

## 📂 Project Structure

```bash
credit-risk-scoring/
├── data/                    # Raw data (not tracked in Git)
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   └── credit_card_balance.csv
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── outputs/                 # Generated artifacts
│   ├── processed_data/      # Cleaned datasets
│   ├── models/              # Saved models
│   └── visualizations/      # Plots and charts
├── docs/                    # Documentation
│   ├── project_report.md
│   └── EDA_findings.md
├── .gitignore
├── README.md
└── requirements.txt
```
## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-scoring.git
cd credit-risk-scoring
```
2. Install dependencies:

```bash
pip install -r requirements.txt
Place your raw data files in the data/ directory

Usage
Run the full pipeline:

python
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_all_features
from src.modeling import train_model

# Load and clean data
data = load_and_clean_data('data')

# Feature engineering
features = create_all_features(data)

# Train model
model, auc = train_model(features, target)
print(f"Model trained with AUC: {auc:.3f}")
```
## 📊 Results

### Model Performance
| Model               | AUC-ROC | Precision | Recall |
|---------------------|---------|-----------|--------|
| LightGBM            | 0.784   | 0.72      | 0.68   |
| Random Forest       | 0.761   | 0.70      | 0.65   |
| Logistic Regression | 0.731   | 0.68      | 0.62   |

### Feature Importance
![Feature Importance](outputs/visualizations/feature_importance.png)

## 🛠 Development

### Running Tests
```bash
python -m pytest tests/
```
Code Style
This project uses:

Black for code formatting

Flake8 for linting

Format your code with:

```bash
black src/
```
## 🤝 Contributors
- [Mohammad Mamun](https://github.com/mamundu509)
- Industry Partner: Nick Jonker (ANZ)
- Academic Advisor: Dr. Yan Wang

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References
- [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)