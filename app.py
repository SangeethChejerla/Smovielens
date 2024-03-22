from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import KNNBasic, Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import requests  # Import requests library for API calls

app = Flask(__name__)

# Load data
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Preprocess movie data
movies_ = movies.copy()
movies_["description"] = movies_["genres"].apply(lambda x: x.replace("|", " "))
movies_ = movies_.drop("genres", axis=1)

# Vectorize movie descriptions
count_v = CountVectorizer()
X_train = count_v.fit_transform(movies_["description"])

# Train KNN model for content-based filtering
neighbor = NearestNeighbors(n_neighbors=10)
neighbor.fit(X_train)

# Train SVD algorithm
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()
algo_svd = SVD(n_factors=20, n_epochs=20)
algo_svd.fit(trainset)


TMDB_API_KEY = "e65f96397db5471ad7bab643b6f327ca"  # Replace with your TMDB API key


# Hybrid recommendation function with TMDB API integration and error handling
def hybrid_recommendation(movie_title):
    movie_index = movies_[movies_["title"] == movie_title].index[0]
    data_for_pred = X_train[movie_index]
    predict = neighbor.kneighbors(data_for_pred, return_distance=True)
    similar_movies = predict[1][0]

    recommendations = []
    for movie_index in similar_movies:
        iid = movies_.iloc[movie_index]["movieId"]
        prediction = algo_svd.predict(1, iid)
        recommendations.append((movies_.iloc[movie_index]["title"], prediction.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    tmdb_recommendations = []
    for title, _ in recommendations[:12]:
        # Strip the year from the movie title
        title_without_year = title.split("(")[0].strip()

        response = requests.get(
            f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_without_year}"
        )
        data = response.json()
        if data.get("results"):  # Check if "results" key exists
            for result in data["results"]:
                movie_id = result["id"]
                details_response = requests.get(
                    f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
                )
                details_data = details_response.json()
                if details_data.get("poster_path") and details_data.get("overview"):
                    tmdb_recommendations.append(
                        {
                            "title": title,
                            "image": f"https://image.tmdb.org/t/p/w500{details_data.get('poster_path')}",
                            "overview": details_data.get("overview"),
                            "genre": details_data.get("genres", [{"name": ""}])[0][
                                "name"
                            ],
                        }
                    )
                    break  # Stop looping if details are found
        else:
            tmdb_recommendations.append(
                {
                    "title": title,
                    "image": "",  # Set placeholder image or message
                    "overview": "No matching movie found on TMDB.",
                    "genre": "",
                }
            )

    return tmdb_recommendations


# Define routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        recommended_movies = hybrid_recommendation(movie_title)
        return render_template(
            "recommendations.html",
            movie_title=movie_title,
            recommended_movies=recommended_movies,
        )
    else:
        return render_template("index.html")


if __name__ == "__main__":
    # Use 0.0.0.0 as the host to listen on all available network interfaces
    app.run(host="0.0.0.0", port=5000, debug=True)
