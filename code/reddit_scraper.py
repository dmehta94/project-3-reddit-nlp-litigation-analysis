# Import pandas for DataFrame handling
import pandas as pd
# Import PRAW to ping the reddit API and handle data
import praw
# Import os
import os
# Import time to handle rate limit
import time
# Import dotenv to load from .env
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Initialize PRAW
reddit = praw.Reddit(
    client_id = os.getenv('REDDIT_ID'),
    client_secret = os.getenv('REDDIT_PASSWORD'),
    user_agent = 'Cybercursor'
)

# Function to retrieve the top posts from a timeframe in a subreddit
def get_top_data(subreddit, time_filter, limit):
    '''
    Retrieve the top [limit] posts from [time_filter] in r/[subreddit].
    Store the following information in a PANDAS DataFrame
    - Timestamp when the post was created
    - Title
    - Post body
    - Subreddit from which the post was collected
    '''
    posts = reddit.subreddit(subreddit).top(time_filter = time_filter, limit = limit)
    data = []
    for post in posts:
        data.append([post.created_utc, post.title, post.selftext, post.subreddit])
    return pd.DataFrame(data, columns = ['created_utc', 'title', 'self_text', 'subreddit'])

# Fucntion to retrieve the newest posts from a timeframe in a subreddit
def get_new_data(subreddit, limit):
    '''
    Retrieve the newest [limit] posts from r/[subreddit].
    Store the following information in a PANDAS DataFrame
    - Timestamp when the post was created
    - Title
    - Post body
    - Subreddit from which the post was collected
    '''
    posts = reddit.subreddit(subreddit).new(limit = limit)
    data = []
    for post in posts:
        data.append([post.created_utc, post.title, post.selftext, post.subreddit])
    return pd.DataFrame(data, columns = ['created_utc', 'title', 'self_text', 'subreddit'])