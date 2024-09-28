from flask import Flask, request, render_template
import pandas as pd
from models.movie_recommendations import generate_recommendations

app = Flask(__name__)

movies = pd.read_csv('./data/movies.csv')
users = pd.read_csv('./data/users.csv')
ratings = pd.read_csv('./data/ratings.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_gender = request.form['gender']
    user_age = int(request.form['age'])

    # Call the movie_recommendations function
    recommendations = generate_recommendations(movies, users, ratings, user_gender, user_age)

    return render_template('recommendation.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
