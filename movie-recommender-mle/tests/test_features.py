import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.user_features import compute_user_features
from features.movie_features import compute_movie_features, ALL_GENRES


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        'userId':    [1, 1, 1, 2, 2, 3],
        'movieId':   [1, 2, 3, 1, 2, 1],
        'rating':    [4.0, 3.5, 5.0, 2.0, 1.5, 4.5],
        'timestamp': [1000, 2000, 3000, 1500, 2500, 1200]
    })

@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        'movieId': [1, 2, 3],
        'title':   ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)'],
        'genres':  ['Adventure|Animation', 'Adventure|Comedy', 'Action|Crime|Thriller']
    })


# ── user feature tests ───────────────────────────────────────────────────────

def test_user_features_row_count(sample_ratings, sample_movies):
    """one row per unique user"""
    result = compute_user_features(sample_ratings, sample_movies)
    assert len(result) == sample_ratings['userId'].nunique()

def test_user_features_no_nulls(sample_ratings, sample_movies):
    """no NaN values in final output"""
    result = compute_user_features(sample_ratings, sample_movies)
    assert result.isnull().sum().sum() == 0

def test_user_features_single_rating_std_is_zero(sample_ratings, sample_movies):
    """user with 1 rating gets std=0, not NaN"""
    result = compute_user_features(sample_ratings, sample_movies)
    user3 = result[result['userId'] == 3]
    assert user3['rating_std'].values[0] == 0.0

def test_user_features_avg_rating_range(sample_ratings, sample_movies):
    """avg rating must be between 0.5 and 5.0 (MovieLens range)"""
    result = compute_user_features(sample_ratings, sample_movies)
    assert result['avg_rating_given'].between(0.5, 5.0).all()

def test_user_features_no_raw_timestamps(sample_ratings, sample_movies):
    """raw timestamps must be dropped from final output"""
    result = compute_user_features(sample_ratings, sample_movies)
    assert 'last_rating_timestamp' not in result.columns
    assert 'first_rating_timestamp' not in result.columns

def test_user_features_days_since_non_negative(sample_ratings, sample_movies):
    """days since last rating can't be negative"""
    result = compute_user_features(sample_ratings, sample_movies)
    assert (result['days_since_last_rating'] >= 0).all()


# ── movie feature tests ──────────────────────────────────────────────────────

def test_movie_features_row_count(sample_ratings, sample_movies):
    """one row per unique movie that has ratings"""
    result = compute_movie_features(sample_ratings, sample_movies)
    assert len(result) == sample_ratings['movieId'].nunique()

def test_movie_features_no_nulls(sample_ratings, sample_movies):
    """no NaN values in final output"""
    result = compute_movie_features(sample_ratings, sample_movies)
    assert result.isnull().sum().sum() == 0

def test_genre_vector_length(sample_ratings, sample_movies):
    """genre columns must equal ALL_GENRES length exactly"""
    result = compute_movie_features(sample_ratings, sample_movies)
    genre_cols = [c for c in result.columns if c.startswith('genre_')]
    assert len(genre_cols) == len(ALL_GENRES)

def test_genre_vector_binary(sample_ratings, sample_movies):
    """genre values must be exactly 0 or 1"""
    result = compute_movie_features(sample_ratings, sample_movies)
    genre_cols = [c for c in result.columns if c.startswith('genre_')]
    assert result[genre_cols].isin([0, 1]).all().all()

def test_release_year_extraction(sample_ratings, sample_movies):
    """release year correctly extracted from title"""
    result = compute_movie_features(sample_ratings, sample_movies)
    assert (result['release_year'] == 1995).all()

def test_raw_genres_column_dropped(sample_ratings, sample_movies):
    """raw genres string column must not appear in output"""
    result = compute_movie_features(sample_ratings, sample_movies)
    assert 'genres' not in result.columns

def test_avg_rating_received_correct(sample_ratings, sample_movies):
    """spot check avg rating for movie 1"""
    result = compute_movie_features(sample_ratings, sample_movies)
    movie1 = result[result['movieId'] == 1]
    expected = sample_ratings[sample_ratings['movieId'] == 1]['rating'].mean()
    assert abs(movie1['avg_rating_received'].values[0] - expected) < 0.001