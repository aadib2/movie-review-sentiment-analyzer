import pandas as pd
import os
import zipfile

def extract_data():
    '''
    Extracts csv dataset from zip file and stores in data directory
    '''
    
    zip_path = "data/imdb-dataset-of-50k-movie-reviews.zip"
    extract_path = 'data'

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
if __name__ == "__main__":
    extract_data()

    
