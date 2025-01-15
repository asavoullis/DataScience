# Diamond Price Prediction

## Overview
This project focuses on building and evaluating machine learning models to predict diamond prices based on various features such as carat, cut, color, clarity, and dimensions (x, y, z). The dataset used is cleaned and preprocessed to ensure reliable results, and multiple regression models are implemented and compared to find the best-performing model.

---

## Features and Dataset
The dataset includes the following features:
- **Carat**: Weight of the diamond.
- **Cut**: Quality of the diamond cut (e.g., Ideal, Premium).
- **Color**: Diamond color, with values ranging from D (best) to J (worst).
- **Clarity**: A measure of diamond clarity (e.g., SI1, VS1).
- **Dimensions (x, y, z)**: Length, width, and depth of the diamond in millimeters.
- **Price**: Target variable representing the price of the diamond (in USD).

---

## Methods

### Preprocessing
1. **Handling Categorical Features**:
   - Label encoding is applied to columns with categorical data (e.g., `cut`, `color`, `clarity`).
2. **Removing Anomalies**:
   - Diamonds with zero dimensions (x, y, z) are filtered out.
3. **Feature Scaling**:
   - StandardScaler is used to standardize the features for better performance with certain regression algorithms.

### Initial Use of PySpark
PySpark was initially explored for preprocessing the dataset, particularly for handling large-scale data operations. However, the dataset size was manageable using pandas, so scikit-learn pipelines were ultimately used for simplicity and integration.

### Model Building
The following regression models are implemented using scikit-learn and XGBoost:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor**
- **XGBoost Regressor**

### Evaluation Metrics
The models are evaluated using the following metrics:
- **R² (Coefficient of Determination)**: Measures the proportion of variance explained by the model.
- **Adjusted R²**: Adjusts R² for the number of predictors.
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values.
- **RMSE (Root Mean Squared Error)**: Square root of MSE for error in the same units as the target variable.

---

## Results
After training and evaluating the models, the XGBoost Regressor achieved the best performance:

- **R²**: `0.9807`
- **Adjusted R²**: `0.9807`
- **MAE**: `$275.90`
- **MSE**: `$306,372.15`
- **RMSE**: `$553.51`

---

## Installation
To run this project, ensure you have the following libraries installed:
- **Python** (>= 3.7)
- **pandas**
- **scikit-learn**
- **xgboost**
- **matplotlib**

Install the required dependencies:
```bash
pip install pandas scikit-learn xgboost matplotlib
```

---

## Usage
1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd diamond-price-prediction
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

---

## Future Work
- Incorporate additional features, such as diamond certification.
- Experiment with other machine learning algorithms (e.g., SVR, Neural Networks).
- Perform hyperparameter tuning for optimal performance.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/).

