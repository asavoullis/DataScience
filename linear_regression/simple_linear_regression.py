# Import necessary libraries
import numpy as np

# Generating random data for features (X)
X = np.random.rand(100, 1) * 10  # Random numbers between 0 and 10

# Generating target values (y) with a linear relationship and some noise
y = 2.5 * X + np.random.randn(100, 1) * 2  # y = 2.5X + noise

from sklearn.model_selection import train_test_split

# Splitting the data: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


import matplotlib.pyplot as plt

# Scatter plot of actual values
plt.scatter(X_test, y_test, color="blue", label="Actual values")
# Line plot of predicted values
plt.plot(X_test, y_pred, color="red", label="Predicted values")
# Adding labels and title
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()


# New data to predict
new_data = np.array([[5], [7], [9]])

# Make predictions
predictions = model.predict(new_data)
print("Predictions for new data:", predictions)
