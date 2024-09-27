# Step 1: Import the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Create a simple dataset
# Here, we define our features (age and movie rating) and labels (like or dislike)
# Features: [age, rating]
X = np.array([
    [22, 8],  # Age 22, Rating 8
    [25, 9],  # Age 25, Rating 9
    [47, 5],  # Age 47, Rating 5
    [51, 6],  # Age 51, Rating 6
    [23, 7],  # Age 23, Rating 7
    [45, 4],  # Age 45, Rating 4
    [29, 10], # Age 29, Rating 10
    [32, 3],  # Age 32, Rating 3
])

# Labels: 1 means they liked the movie, 0 means they did not
y = np.array([1, 1, 0, 0, 1, 0, 1, 0])

# Step 3: Split the dataset into training and testing sets
# This helps us evaluate how well our model works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Create and train the model
model = LogisticRegression()  # Create a logistic regression model
model.fit(X_train, y_train)    # Train the model with our training data

# Step 5: Make predictions
predictions = model.predict(X_test)  # Use the model to predict on the test data

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, predictions)  # Calculate the accuracy
print(f'Predictions: {predictions}')
print(f'Accuracy: {accuracy:.2f}')  # Print the accuracy
