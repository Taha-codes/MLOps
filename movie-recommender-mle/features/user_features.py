import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_user_features(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user features from ratings history.
    
    These features will be stored in both:
    - Parquet (offline store) for training
    - Redis (online store) for serving
    
    Using the same function for both prevents training-serving skew.
    """
    print("Computing user features...")

    # Merge to get genre info
    ratings_with_genres = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

    # Base aggregations
    user_features = ratings.groupby('userId').agg(
        avg_rating_given=('rating', 'mean'), # understaing user rating bias
        rating_std=('rating', 'std'), # strong opinions of users
        total_ratings=('rating', 'count'), # users with more ratings are more cricual
        last_rating_timestamp=('timestamp', 'max'), # how recently active is each user
        first_rating_timestamp=('timestamp', 'min') # distinguishes between two users with the same total_ratings
    ).reset_index()

    # Days since last rating (relative to max timestamp in dataset)
    max_timestamp = ratings['timestamp'].max()
    user_features['days_since_last_rating'] = (
        (max_timestamp - user_features['last_rating_timestamp']) / 86400
    ).astype(int)

    # Activity span in days
    user_features['activity_span_days'] = (
        (user_features['last_rating_timestamp'] - user_features['first_rating_timestamp']) / 86400
    ).astype(int)

    # Fill NaN std (users with only 1 rating)
    user_features['rating_std'] = user_features['rating_std'].fillna(0)

    # Drop raw timestamps
    user_features = user_features.drop(
        columns=['last_rating_timestamp', 'first_rating_timestamp']
    )

    print(f"Computed features for {len(user_features)} users")
    return user_features

if __name__ == "__main__":
    ratings = pd.read_csv('/Users/mac/Desktop/MLOps/movie-recommender-mle/data/raw/ml-25m/ratings.csv')
    movies = pd.read_csv('/Users/mac/Desktop/MLOps/movie-recommender-mle/data/raw/ml-25m/movies.csv')

    user_features = compute_user_features(ratings, movies)
    print(user_features.head(20))
    print(user_features.dtypes)