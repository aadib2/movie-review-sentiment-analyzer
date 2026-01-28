# necessary imports
import nltk
import string
import re  # regular expression

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# necessary packages to use imports above
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class Cleaner:
    def __init__(self, keep_negations=True, min_token_length=2):
        """
        Initialize the Cleaner with preprocessing configuration.
        
        Args:
            keep_negations (bool): whether to preserve negation words
            min_token_length (int): minimum token length to keep
        """
        self.keep_negations = keep_negations
        self.min_token_length = min_token_length
        
        # define stopwords, punctuation to remove
        self.stopw = set(stopwords.words('english'))  # includes common words in reviews
        # keep any negation words since they are important for sentiment
        self.stopw = self.stopw - {'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing'}
        
        self.punctuation = re.escape(string.punctuation)  # originally was using list but this is cleaner and faster, uses regular expressions
        
        # define the lemmatizer from nltk (reduces words to their base form like acting-> act)
        self.lemmatizer = WordNetLemmatizer()
        
        # compile regex patterns once (more efficient)
        self.html_pattern = re.compile(r'<.*?>')
        self.punctuation_pattern = re.compile(rf'[{self.punctuation}]')
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        
        # contraction patterns
        self.contractions = {
            r"n't": " not",
            r"'m": " am",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would"
        }
    
    def _handle_encoding_issues(self, document):
        """Handle common encoding issues in text."""
        document = document.replace('\x96', '—')  # em dash
        document = document.replace('\x91', "'")   # Left single quote
        document = document.replace('\x92', "'")   # Right single quote
        document = document.replace('\x93', '"')   # Left double quote
        document = document.replace('\x94', '"')   # Right double quote
        document = document.replace('\x97', '—')   # Em dash
        return document
    
    def _expand_contractions(self, document):
        """Expand contractions to preserve sentiment-bearing words like 'not'."""
        for pattern, replacement in self.contractions.items():
            document = re.sub(pattern, replacement, document)
        return document
    
    def _remove_html_and_urls(self, document):
        """Remove HTML tags and URLs from text."""
        document = self.html_pattern.sub('', document)
        document = self.url_pattern.sub('', document)
        return document
    
    def _normalize_text(self, document):
        """Normalize text: lowercase, strip whitespace, remove punctuation."""
        document = document.strip().lower()
        document = self.punctuation_pattern.sub(' ', document)
        return document
    
    def _tokenize_and_filter(self, document):
        """Tokenize text and filter out stopwords, short tokens, then lemmatize."""
        tokens = word_tokenize(document)
        
        # Filter: remove stopwords, short tokens, and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stopw and len(word) >= self.min_token_length
        ]
        
        return tokens
    
    def clean(self, document):
        """
        Preprocess text for sentiment analysis using various techniques (regex, stopwords, lemmatize, tokenize)
        
        Args: 
            document (str): The text to preprocess

        Returns: 
            str: The cleaned and preprocessed text, ready for vectorization
        """
        # handle empty documents or non-strings (just incase)
        if not isinstance(document, str):
            return ''
        
        # Process the document through the pipeline
        document = self._handle_encoding_issues(document)
        
        if self.keep_negations:
            document = self._expand_contractions(document)
        
        document = self._remove_html_and_urls(document)
        document = self._normalize_text(document)
        
        tokens = self._tokenize_and_filter(document)
        cleaned_doc = ' '.join(tokens)
        
        return cleaned_doc

