from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

# Load the movie dataset
data = pd.read_csv('data/movies.csv')
data = data.dropna(subset=['title', 'description'])

# Precompute TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(data.index, index=data['title'].str.lower()).drop_duplicates()

# TMDB API setup
TMDB_API_KEY = 'YOUR_TMDB_API_KEY'

def fetch_poster(title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(search_url)
    if response.status_code == 200:
        results = response.json()['results']
        if results:
            poster_path = results[0].get('poster_path')
            return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
    return None

def get_recommendations(title):
    idx = movie_indices.get(title.lower())
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_ids = [i[0] for i in sim_scores]
    recommendations = []
    for i in movie_ids:
        movie = {
            'title': data.iloc[i]['title'],
            'genres': data.iloc[i]['genres'],
            'poster': fetch_poster(data.iloc[i]['title'])
        }
        recommendations.append(movie)
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie = request.form['movie']
        recs = get_recommendations(movie)
        return render_template('recommendations.html', movie=movie, recommendations=recs)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
