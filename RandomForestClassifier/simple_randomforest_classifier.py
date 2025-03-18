# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Features and target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example: Predict the class of a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Example flower features
prediction = rf_classifier.predict(new_flower)
print(f"Predicted class: {iris.target_names[prediction[0]]}")

# Example: Likely a Versicolor or Virginica flower
new_flower2 = [[6.5, 3.0, 5.2, 2.0]]
prediction2 = rf_classifier.predict(new_flower2)
print(f"Predicted class: {iris.target_names[prediction2[0]]}")

print(
    "\nEvaluate the model's predictions on the test data (y_test) to ensure it's correctly classifying multiple species:"
)
print(rf_classifier.predict(X_test))


"""
    -Dataset: We use the Iris dataset, which is preloaded in scikit-learn. 
    It contains three classes of flowers: setosa, versicolor, and virginica.

    -Splitting Data: We divide the data into training and testing sets using train_test_split.

    -Random Forest: The RandomForestClassifier builds multiple decision trees to classify the data.

    -Evaluation: The model's accuracy is checked using the testing set.

    -Prediction: You can use the trained model to predict the class of new data points.

"""
