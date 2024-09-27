import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Use pandas to read data from csv
movies = pd.read_csv('./data/movies.csv')
ratings = pd.read_csv('./data/ratings.csv')

#Merge ratings and movies csv's into one table: data
data = pd.merge(ratings, movies, on='movieId')

#Use numpy to define whether someone liked a movie based on a rating of 4 or higher
data['liked'] = np.where(data['rating'] >= 4.0, 1, 0)

#feature of interest is the mean rating per movieId
features = data.groupby('movieId')['rating'].mean().reset_index()

#fraction of people who liked the movie
labels = (data.groupby('movieId')['liked'].mean() >= 0.5).astype(int).reset_index()

merged_data = pd.merge(features, labels, on='movieId')

X = merged_data[['rating']]
y = merged_data['liked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'Predictions: {predictions}')
print(f'Accuracy: {accuracy:.2f}')

