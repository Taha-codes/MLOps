import pandas as pd
import numpy as np
import redis
import json
import os
from pathlib import Path

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
OFFLINE_STORE_PATH = Path('data/processed')

class FeatureStore:

    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        OFFLINE_STORE_PATH.mkdir(parents=True, exist_ok=True)

    def save_user_features(self, user_features: pd.DataFrame):
        print("Saving user features to offline store (Parquet)...")
        path = OFFLINE_STORE_PATH / 'user_features.parquet'
        user_features.to_parquet(path, index=False)
        print(f"Saved to {path}")

        print("Saving user features to online store (Redis)...")
        self._write_to_redis(user_features, key_prefix='user', id_col='userId')
        print("User features saved to Redis")

    def save_movie_features(self, movie_features: pd.DataFrame):
        print("Saving movie features to offline store (Parquet)...")
        path = OFFLINE_STORE_PATH / 'movie_features.parquet'
        movie_features.to_parquet(path, index=False)
        print(f"Saved to {path}")

        print("Saving movie features to online store (Redis)...")
        self._write_to_redis(movie_features, key_prefix='movie', id_col='movieId')
        print("Movie features saved to Redis")

    def _write_to_redis(self, df: pd.DataFrame, key_prefix: str, id_col: str):
        pipe = self.redis_client.pipeline()
        for _, row in df.iterrows():
            key = f"{key_prefix}:{int(row[id_col])}"
            # Convert to dict, handle numpy types
            value = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in row.items()
            }
            pipe.set(key, json.dumps(value))
        pipe.execute()

    def get_user_features(self, user_id: int) -> dict:
        key = f"user:{user_id}"
        value = self.redis_client.get(key)
        if value is None:
            return {}
        return json.loads(value)

    def get_movie_features(self, movie_id: int) -> dict:
        key = f"movie:{movie_id}"
        value = self.redis_client.get(key)
        if value is None:
            return {}
        return json.loads(value)

    def load_user_features_offline(self) -> pd.DataFrame:
        path = OFFLINE_STORE_PATH / 'user_features.parquet'
        return pd.read_parquet(path)

    def load_movie_features_offline(self) -> pd.DataFrame:
        path = OFFLINE_STORE_PATH / 'movie_features.parquet'
        return pd.read_parquet(path)


if __name__ == "__main__":
    from user_features import compute_user_features
    from movie_features import compute_movie_features

    # Load raw data
    ratings = pd.read_csv('/Users/mac/Desktop/MLOps/movie-recommender-mle/data/raw/ml-25m/ratings.csv')
    movies = pd.read_csv('/Users/mac/Desktop/MLOps/movie-recommender-mle/data/raw/ml-25m/movies.csv')

    # Compute features
    user_features = compute_user_features(ratings, movies)
    movie_features = compute_movie_features(ratings, movies)

    # Save to both stores
    store = FeatureStore()
    store.save_user_features(user_features)
    store.save_movie_features(movie_features)

    # Verify Redis reads work
    print("\nVerifying Redis reads...")
    sample_user = store.get_user_features(1)
    sample_movie = store.get_movie_features(1)
    print(f"User 1 features: {sample_user}")
    print(f"Movie 1 features: {sample_movie}")