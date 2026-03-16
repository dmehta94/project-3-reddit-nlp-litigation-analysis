"""
reddit_scraper.py

Data collection module for the D&D and Pathfinder NLP classification project.
Uses PRAW to retrieve posts from two subreddits and writes them to CSV files
for downstream analysis.

Usage:
    from reddit_scraper import unified_data
    unified_data('DungeonsAndDragons', 'all', 1000, 1000)
"""

import os

import pandas as pd
import praw
from dotenv import load_dotenv

# Resolve and load .env from the same directory as this script
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_ENV_PATH)

# Column names for post data — defined once to avoid magic strings across functions
POST_COLUMNS = ['id', 'created_utc', 'title', 'self_text', 'subreddit']

# Validate credentials before attempting to connect
_CLIENT_ID = os.getenv('REDDIT_ID')
_CLIENT_SECRET = os.getenv('REDDIT_SECRET')

if not _CLIENT_ID or not _CLIENT_SECRET:
    raise EnvironmentError(
        "Reddit API credentials not found. "
        "Ensure REDDIT_ID and REDDIT_SECRET are set in your .env file."
    )

reddit = praw.Reddit(
    client_id=_CLIENT_ID,
    client_secret=_CLIENT_SECRET,
    user_agent='Cybercursor'
)


def get_top_data(subreddit: str, time_filter: str, limit: int) -> pd.DataFrame:
    """
    Retrieve the top posts from a subreddit over a given time window.

    Args:
        subreddit: Name of the subreddit to query (e.g., 'DungeonsAndDragons').
        time_filter: Time window for top posts. One of 'hour', 'day', 'week',
            'month', 'year', or 'all'.
        limit: Maximum number of posts to retrieve. PRAW may return fewer if
            posts have been deleted or are otherwise unavailable.

    Returns:
        DataFrame with columns: id, created_utc, title, self_text, subreddit.

    Example:
        >>> df = get_top_data('DungeonsAndDragons', 'all', 1000)
        >>> df.shape
        (982, 5)
    """
    posts = reddit.subreddit(subreddit).top(time_filter=time_filter, limit=limit)
    records = [
        [post.id, post.created_utc, post.title, post.selftext, post.subreddit]
        for post in posts
    ]
    return pd.DataFrame(records, columns=POST_COLUMNS)


def get_new_data(subreddit: str, limit: int) -> pd.DataFrame:
    """
    Retrieve the most recent posts from a subreddit.

    Args:
        subreddit: Name of the subreddit to query (e.g., 'Pathfinder_RPG').
        limit: Maximum number of posts to retrieve. PRAW may return fewer if
            posts have been deleted or are otherwise unavailable.

    Returns:
        DataFrame with columns: id, created_utc, title, self_text, subreddit.

    Example:
        >>> df = get_new_data('Pathfinder_RPG', 1000)
        >>> df.shape
        (998, 5)
    """
    posts = reddit.subreddit(subreddit).new(limit=limit)
    records = [
        [post.id, post.created_utc, post.title, post.selftext, post.subreddit]
        for post in posts
    ]
    return pd.DataFrame(records, columns=POST_COLUMNS)


def unified_data(
    subreddit: str,
    time_filter: str,
    limit_top: int,
    limit_new: int,
    output_dir: str = '../data'
) -> None:
    """
    Collect top and recent posts from a subreddit and write deduplicated results to CSV.

    Retrieves the top posts for a given time window and the most recent posts,
    concatenates them, removes duplicates by post ID, and writes the result to
    <output_dir>/<subreddit>.csv.

    Args:
        subreddit: Name of the subreddit to query (e.g., 'DungeonsAndDragons').
        time_filter: Time window for top posts. One of 'hour', 'day', 'week',
            'month', 'year', or 'all'.
        limit_top: Maximum number of top posts to retrieve.
        limit_new: Maximum number of recent posts to retrieve.
        output_dir: Directory in which to write the output CSV. Defaults to '../data'.

    Returns:
        None. Writes a CSV file to <output_dir>/<subreddit>.csv.

    Example:
        >>> unified_data('DungeonsAndDragons', 'all', 1000, 1000)
        # Writes deduplicated posts to ../data/DungeonsAndDragons.csv
    """
    top_posts = get_top_data(subreddit, time_filter, limit_top)
    new_posts = get_new_data(subreddit, limit_new)

    unified_posts = (
        pd.concat([top_posts, new_posts])
        .drop_duplicates(subset='id')
        .reset_index(drop=True)
    )

    output_path = os.path.join(output_dir, f'{subreddit}.csv')
    unified_posts.to_csv(output_path, index=False)