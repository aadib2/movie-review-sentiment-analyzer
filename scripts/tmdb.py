import os
from dotenv import load_dotenv
import tmdbsimple as tmdb

load_dotenv()

tmdb_api = os.getenv('TMDB_API_KEY')

tmdb.API_KEY = tmdb_api

# a test case for the movie 'The Matrix'
review = tmdb.Reviews()
response = movie.info()

print(movie.title)
print(movie.budget)
print(response)