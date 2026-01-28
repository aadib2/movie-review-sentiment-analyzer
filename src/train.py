import os
from preprocessing import Cleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time # for measuring training time

class Trainer:
    def __init__(self):
        self.config = self.load_config()

    
    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
    
    def create_pipeline(self):
        # init cleaner

        # init vectorizer & label encoder

        # init model and params from config

        # define Pipeline()

        # return pipeline
        return