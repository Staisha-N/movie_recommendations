import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

movies = pd.read_csv('./data/movies.csv')
users = pd.read_csv('./data/users.csv')
ratings = pd.read_csv('./data/ratings.csv')

user_gender = 'F'
user_age = 30
filtered_users = users[(users['gender'] == user_gender) & 
                       (users['age'] >= user_age - 10) & 
                       (users['age'] <= user_age + 10)]
filtered_ratings = ratings[ratings['userId'].isin(filtered_users['userId'])]

# Prepare data for PyTorch
class MovieRatingsDataset(Dataset):
    def __init__(self, ratings):
        self.user_ids = ratings['userId'].values
        self.movie_ids = ratings['movieId'].values
        self.ratings = ratings['rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.movie_ids[idx]), self.ratings[idx]

# Define neural network
class RatingPredictor(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=8):
        super(RatingPredictor, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_id, movie_id):
        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)
        x = torch.cat([user_emb, movie_emb], dim=1)
        return self.fc(x)

# Prepare data
num_users = ratings['userId'].nunique() + 1  # +1 for zero-index
num_movies = ratings['movieId'].nunique() + 1  # +1 for zero-index
dataset = MovieRatingsDataset(filtered_ratings)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function and optimizer
model = RatingPredictor(num_users, num_movies)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(10):  # You can increase the number of epochs
    for (user_ids, movie_ids), ratings in dataloader:
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids).squeeze()
        loss = criterion(predictions, ratings.float())
        loss.backward()
        optimizer.step()

# Generating recommendations
model.eval()
user_id = torch.tensor([filtered_users['userId'].iloc[0]])  # Replace with actual userId
movie_ids = torch.tensor(range(1, num_movies))  # Assuming movieId starts from 1
predicted_ratings = model(user_id.repeat(len(movie_ids)), movie_ids).detach().numpy()

# Create DataFrame for recommendations
recommended_movies = pd.DataFrame({
    'movieId': movie_ids.numpy(),
    'predicted_rating': predicted_ratings.flatten()  # Flatten the predictions
}).merge(movies, on='movieId')

# Get top recommendations
top_recommendations = recommended_movies.nlargest(2, 'predicted_rating')

print("Top 2 Recommendations:")
for index, row in top_recommendations.iterrows():
    print(f"Title: {row['title']}, Predicted Rating: {row['predicted_rating']:.2f}")
