# Import pandas for DataFrame handling
import pandas as pd
# Import PRAW to ping the reddit API and handle data
import praw
# Import os
import os
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

# Choose subreddits
r_dungeons_and_dragons = reddit.subreddit('DungeonsAndDragons')
r_pathfinder_rpg = reddit.subreddit('Pathfinder_RPG')

# Pull post data from subreddits
dnd_posts = r_dungeons_and_dragons.top(time_filter = 'all', limit = 1000)
pathfinder_posts = r_pathfinder_rpg.top(time_filter = 'all', limit = 1000)

# Write posts to lists of data
dnd_data = []
for post in dnd_posts:
    dnd_data.append([post.created_utc, post.title, post.selftext, post.subreddit])

pathfinder_data = []
for post in pathfinder_posts:
    pathfinder_data.append([post.created_utc, post.title, post.selftext, post.subreddit])

# Convert to Pandas DataFrames
dnd = pd.DataFrame(dnd_data, columns = ['created_utc', 'title', 'self_text', 'subreddit'])
pathfinder = pd.DataFrame(pathfinder_data, columns = ['created_utc', 'title', 'self_text', 'subreddit'])

# Write data to files
dnd.to_csv('../data/dnd.csv')
pathfinder.to_csv('../data/pathfinder.csv')