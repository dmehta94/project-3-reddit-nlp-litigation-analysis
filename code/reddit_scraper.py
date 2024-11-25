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
    Retrieve the top [limit] posts of [time_filter] from r/[subreddit].
    Store the following information in a PANDAS DataFrame
    - Post ID
    - Timestamp when the post was created
    - Title
    - Post body
    - Subreddit from which the post was collected
    '''
    posts = reddit.subreddit(subreddit).top(time_filter = time_filter, limit = limit)
    data = []
    for post in posts:
        data.append([post.id, post.created_utc, post.title, post.selftext, post.subreddit])
    return pd.DataFrame(data, columns = ['id', 'created_utc', 'title', 'self_text', 'subreddit'])

# Fucntion to retrieve the newest posts from a timeframe in a subreddit
def get_new_data(subreddit, limit):
    '''
    Retrieve the newest [limit] posts from r/[subreddit].
    Store the following information in a PANDAS DataFrame
    - Post ID
    - Timestamp when the post was created
    - Title
    - Post body
    - Subreddit from which the post was collected
    '''
    posts = reddit.subreddit(subreddit).new(limit = limit)
    data = []
    for post in posts:
        data.append([post.id, post.created_utc, post.title, post.selftext, post.subreddit])
    return pd.DataFrame(data, columns = ['id', 'created_utc', 'title', 'self_text', 'subreddit'])

def unified_data(subreddit, time_filter, limit_top, limit_new):
    '''
    Retrieve the top [limit_top] posts of [time_filter]
    and the newest [limit_new] posts from r/[subreddit],
    then concatenate them to the same PANDAS DataFrame
    to be saved as a .csv file in ../data/.
    '''
    top_posts = get_top_data(subreddit, time_filter, limit_top)
    new_posts = get_new_data(subreddit, limit_new)
    unified_posts = pd.concat([top_posts, new_posts]).drop_duplicates(subset = 'id').reset_index(drop = True)
    return unified_posts.to_csv(f'../data/{subreddit}.csv', index = False)