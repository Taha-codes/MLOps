import pandas as pd
import numpy as np
import re

# All genres in MovieLens dataset
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western', 'IMAX'
]

def extract_release_year(title: str) -> int:
    """Extract release year from movie title like 'Toy Story (1995)'"""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else 0

def compute_genre_vector(genres_str: str) -> list:
    """Convert genre string to multi-hot encoded vector"""
    genres = genres_str.split('|')
    return [1 if g in genres else 0 for g in ALL_GENRES]

def compute_movie_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-movie features from ratings received.

    Same principle as user features — one function,
    used for both training and serving.
    """
    print("Computing movie features...")

    # Base aggregations from ratings
    movie_features = ratings.groupby('movieId').agg(
        avg_rating_received=('rating', 'mean'),
        rating_std=('rating', 'std'),
        total_ratings=('rating', 'count')
    ).reset_index()

    # Merge with movie metadata
    movie_features = movie_features.merge(movies, on='movieId', how='left')

    # Extract release year from title
    movie_features['release_year'] = movie_features['title'].apply(extract_release_year)

    # Compute genre vectors
    genre_vectors = movie_features['genres'].apply(compute_genre_vector)
    genre_df = pd.DataFrame(
        genre_vectors.tolist(),
        columns=[f'genre_{g.lower().replace("-", "_")}' for g in ALL_GENRES]
    )
    movie_features = pd.concat([movie_features, genre_df], axis=1)

    # Fill NaN std (movies with only 1 rating)
    movie_features['rating_std'] = movie_features['rating_std'].fillna(0)

    # Drop raw genre string and title (we have structured versions now)
    movie_features = movie_features.drop(columns=['genres'])

    print(f"Computed features for {len(movie_features)} movies")
    return movie_features

if __name__ == "__main__":
    ratings = pd.read_csv('data/raw/ml-25m/ratings.csv')
    movies = pd.read_csv('data/raw/ml-25m/movies.csv')

    movie_features = compute_movie_features(ratings, movies)
    print(movie_features.head())
    print(movie_features.dtypes)