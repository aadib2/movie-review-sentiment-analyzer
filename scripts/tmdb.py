import os
from dotenv import load_dotenv
import tmdbsimple as tmdb

load_dotenv()

tmdb_api = os.getenv('TMDB_API_KEY')

tmdb.API_KEY = tmdb_api

# a test case for the movie 'The Matrix'
# reviews = tmdb.Movies(id=603).reviews()
# print(reviews['results'][0]['content'])

# get popular movies
reviews_avatar = tmdb.Movies(id=83533).reviews()
print(reviews_avatar)
