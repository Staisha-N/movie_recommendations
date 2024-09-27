import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

movies = pd.read_csv('./data/movies.csv')
users = pd.read_csv('./data/users.csv')
ratings = pd.read_csv('./data/ratings.csv')

user_gender = 'F'
user_age = 30

filtered_users = users[(users['gender'] == user_gender) & (users['age'] >= user_age - 10) & (users['age'] <= user_age + 10)]
filtered_ratings = ratings[ratings['userId'].isin(filtered_users['userId'])]
average_ratings = filtered_ratings.groupby('movieId')['rating'].mean().reset_index()
recommended_movies = average_ratings.merge(movies, on='movieId')
top_recommendations = recommended_movies.nlargest(2, 'rating')

print("Top 2 Recommendations:")
for index, row in top_recommendations.iterrows():
    print(f"Title: {row['title']}, Average Rating: {row['rating']:.2f}")
