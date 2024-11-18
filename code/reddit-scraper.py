# Import PRAW to ping the reddit API and handle data
import praw
# Import .getenv
from os import getenv


reddit = praw.Reddit(
    client_id = getenv('REDDIT_ID'),
    client_secret = getenv('REDDIT_PASSWORD')
)