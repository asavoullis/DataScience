import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data with more points and variance
np.random.seed(42)
X = 2 * np.random.rand(500, 1)
# y = 4 + 3 * X + np.random.randn(500, 1) * 1.5  # Adding more variance
y = 4 + 3 * X + np.random.randn(500, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compute residuals
residuals = y_test - y_pred

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
variance = np.var(y_pred)

# Print model evaluation metrics
print("")
print(f"Coefficients: {model.coef_[0][0]:.2f}")  # Slope of the regression line
print(f"Intercept: {model.intercept_[0]:.2f}")  # Y-intercept of the regression line
print(
    f"Mean Squared Error: {mse:.2f}"
)  # Measures the average squared difference between actual and predicted values
print(
    f"R² Score: {r2:.2f}"
)  # Represents how well the model explains the variance in the data
print(
    f"Variance of Predictions: {variance:.2f}"
)  # Measures the spread of predicted values

# Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Model")
plt.show()

# Plot residuals
plt.figure(figsize=(8, 6))
plt.scatter(X_test, residuals, color="purple", alpha=0.6)
plt.axhline(y=0, color="black", linestyle="--", linewidth=2)
plt.xlabel("X")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.show()

"""

Residuals (Actual vs. Predicted Values)
-   Residuals = Actual values - Predicted values.
-   The residuals plot helps check if errors are randomly distributed (a sign of a good model).

Model Evaluation Metrics
-   Coefficients: Slope of the regression line (how much y changes per unit X).
-   Intercept: Starting point of the regression line when X=0.
-   Mean Squared Error (MSE): Measures how far predictions deviate from actual values (lower is better).
-   R² Score: Explains how much variance in y is explained by X (closer to 1 is better).
-   Variance of Predictions: Measures spread of predicted values (useful for understanding model stability).

"""
