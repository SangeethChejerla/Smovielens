import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
import requests
import asyncio
from aiohttp import ClientSession
import concurrent.futures

# Load data (consider error handling for potential file issues)
try:
    movies = pd.read_csv("dataset/movies.csv")
    ratings = pd.read_csv("dataset/ratings.csv")
except FileNotFoundError:
    print(
        "Error: Data files not found. Please ensure 'dataset/movies.csv' and 'dataset/ratings.csv' exist in the same directory as this script."
    )
    exit()

# Preprocess movie data (cache for efficiency)
movies["description"] = movies["genres"].apply(lambda x: x.replace("|", " "))
movies = movies.drop("genres", axis=1)

# Vectorize movie descriptions with TF-IDF for better weighting
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(movies["description"])

# Train KNN model
neighbor = NearestNeighbors(n_neighbors=10, metric="euclidean")
neighbor.fit(X_train)

# Train SVD algorithm
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()
algo_svd = SVD(n_factors=20, n_epochs=20)
algo_svd.fit(trainset)

TMDB_API_KEY = "e65f96397db5471ad7bab643b6f327ca"  # Replace with your TMDB API key


async def fetch_movie_details(session, title_without_year):
    try:
        async with session.get(
            f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_without_year}"
        ) as response:
            data = await response.json()
            if data.get("results"):
                result = data["results"][0]
                movie_id = result["id"]
                async with session.get(
                    f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
                ) as details_response:
                    details_data = await details_response.json()
                    if details_data.get("poster_path") and details_data.get("overview"):
                        return {
                            "title": title_without_year,
                            "image": f"https://image.tmdb.org/t/p/w500{details_data.get('poster_path')}",
                            "overview": details_data.get("overview"),
                            "genre": details_data.get("genres", [{"name": ""}])[0][
                                "name"
                            ],
                        }
    except Exception as e:
        print(f"Error fetching movie details: {e}")
    return None


async def get_movie_recommendations(movie_title):
    movie_index = movies[movies["title"] == movie_title].index[0]
    data_for_pred = X_train[movie_index]
    predict = neighbor.kneighbors(data_for_pred, return_distance=True)
    similar_movies = predict[1][0]

    recommendations = []
    for movie_index in similar_movies:
        iid = movies.iloc[movie_index]["movieId"]
        prediction = algo_svd.predict(1, iid)
        recommendations.append((movies.iloc[movie_index]["title"], prediction.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    async with ClientSession() as session:
        tasks = []
        for title, _ in recommendations[:12]:
            tasks.append(fetch_movie_details(session, title.split("(")[0].strip()))
        results = await asyncio.gather(*tasks)
        return [result for result in results if result]


# Flask app setup
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# Define routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    movie_title = request.form.get("movie_title")
    if not movie_title:
        return jsonify({"error": "Movie title is required"}), 400

    recommended_movies = asyncio.run(get_movie_recommendations(movie_title))
    return render_template(
        "recommendations.html",
        movie_title=movie_title,
        recommended_movies=recommended_movies,
    )


# fix
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
