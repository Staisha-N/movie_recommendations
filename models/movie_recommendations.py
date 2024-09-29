import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.genre_convert import transform_genres

class MovieRatingsDataset(Dataset):
    def __init__(self, ratings, age_genre_relations, user_age):
        self.user_ids = ratings['userId'].values
        self.movie_ids = ratings['movieId'].values
        self.ratings = ratings['rating'].values
        self.age_genre_features = self.get_age_genre_features(age_genre_relations, user_age)

    def get_age_genre_features(self, age_genre_relations, user_age):
        age_group = age_genre_relations[age_genre_relations['age'] == user_age].iloc[0, 1:].values
        return [age_group] * len(self.ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.movie_ids[idx], self.age_genre_features[idx]), self.ratings[idx]

class RatingPredictor(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=8, genre_dim=4):
        super(RatingPredictor, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_fc = nn.Linear(genre_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 3, 1)

    def forward(self, user_id, movie_id, genre_features):
        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)
        genre_emb = self.genre_fc(genre_features)
        x = torch.cat([user_emb, movie_emb, genre_emb], dim=1)
        return self.fc(x)

def generate_recommendations(movies, users, ratings, user_gender, user_age):
    transform_genres("./data/age_genre_relations.csv", "./data/transformed_age_genre_relations.csv")
    age_genre_relations = pd.read_csv('./data/transformed_age_genre_relations.csv')

    filtered_users = users[(users['gender'] == user_gender) & 
                            (users['age'] >= user_age - 10) & 
                            (users['age'] <= user_age + 10)]
    filtered_ratings = ratings[ratings['userId'].isin(filtered_users['userId'])]

    # Prepare data
    num_users = ratings['userId'].nunique() + 1
    num_movies = ratings['movieId'].nunique() + 1
    dataset = MovieRatingsDataset(filtered_ratings, age_genre_relations, user_age)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model, loss function and optimizer
    model = RatingPredictor(num_users, num_movies)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(10):
        for (user_ids, movie_ids, genre_features), ratings in dataloader:
            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids, genre_features.float()).squeeze()
            loss = criterion(predictions, ratings.float())
            loss.backward()
            optimizer.step()

    # Generating recommendations
    model.eval()
    user_id = torch.tensor([filtered_users['userId'].iloc[0]])  # Use the first matched user
    movie_ids = torch.tensor(range(1, num_movies))
    genre_features = torch.tensor([age_genre_relations.loc[age_genre_relations['age'] == user_age].iloc[0, 1:].values] * len(movie_ids))
    predicted_ratings = model(user_id.repeat(len(movie_ids)), movie_ids, genre_features.float()).detach().numpy()

    # Create DataFrame for recommendations
    recommended_movies = pd.DataFrame({
        'movieId': movie_ids.numpy(),
        'predicted_rating': predicted_ratings.flatten()
    }).merge(movies, on='movieId')

    # Get top recommendations
    top_recommendations = recommended_movies.nlargest(2, 'predicted_rating')
    return top_recommendations[['title', 'predicted_rating']].to_dict(orient='records')