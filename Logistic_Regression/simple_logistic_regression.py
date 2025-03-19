import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate synthetic binary classification data
np.random.seed(42)
X = 2 * np.random.rand(500, 1)
y = (
    (4 + 3 * X + np.random.randn(500, 1) * 1.5 > 6).astype(int).ravel()
)  # Binary labels (0 or 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")  # Measures the proportion of correct predictions
print("Confusion Matrix:")
print(
    conf_matrix
)  # Shows True Positives, False Positives, False Negatives, and True Negatives
print("Classification Report:")
print(class_report)  # Precision, Recall, F1-Score, and Support for each class

# Plot logistic regression curve
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_prob, color="red", label="Predicted Probability", alpha=0.6)
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.title("Logistic Regression Model")
plt.show()


"""
Logistic regression is defined as a supervised machine learning algorithm 
that accomplishes BINARY CLASIFICATION tasks by predicting the probability 
of an outcome, event, or observation.

The model delivers a binary or dichotomous outcome limited to two possible outcomes: 
yes/no, 0/1, or true/false.

Logical regression analyzes the relationship between one or more independent variables
and classifies data into discrete classes. It is extensively used in predictive modeling,
where the model estimates the mathematical probability of whether an instance belongs to a specific category or not.

For example, 0 - represents a negative class; 1 - represents a positive class. 
Logistic regression is commonly used in binary classification problems where 
the outcome variable reveals either of the two categories (0 and 1).


Data Generation
    -We create synthetic data where y is a binary variable (0 or 1).
    -The condition (4 + 3 * X + noise > 6) defines a decision boundary.

Data Splitting
    -Train-Test Split ensures the model generalizes well.
    -80% for training, 20% for testing.

Model Training
    -Logistic Regression is trained using fit(X_train, y_train).
    -It learns the sigmoid relationship between X and y.

Predictions
    -predict(X_test): Returns class labels (0 or 1).
    -predict_proba(X_test): Returns probabilities for class 1 (used for threshold-based decisions).

Model Evaluation
    -Accuracy Score: Measures how many predictions are correct.
    -Confusion Matrix: Shows True Positives, False Positives, False Negatives, True Negatives.
    -Classification Report: Provides Precision, Recall, and F1-Score.

Visualization
    -The logistic regression curve shows actual points vs. predicted probabilities.

    
www.spiceworks.com/tech/artificial-intelligence/articles/what-is-logistic-regression/
    

"""
