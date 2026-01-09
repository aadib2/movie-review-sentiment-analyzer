import nltk
import string
import re # regular expression

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# necessary packages to use imports above
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# define stopwords, punctuation to remove
stopw = set(stopwords.words('english')) # includes common words in reviews
# keep any negation words since they are important for sentiment
stopw = stopw - {'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing'}
punctuation = re.escape(string.punctuation) # originally was using list but this is cleaner and faster, uses regular expressions

# define the lemmatizer from nltk (reduces words to their base form like acting-> act)
lemmatizer = WordNetLemmatizer()

# compile regex patterns once (more efficient)
html_pattern = re.compile(r'<.*?>')
punctuation_pattern = re.compile(rf'[{punctuation}]')
url_pattern = re.compile(r'http\S+|www\.\S+')

# define preprocessing pipeline in method
def preprocess(document, keep_negations=True, min_token_length=2):
    '''
    Preprocess text for sentiment analysis using various techniques (regex, stopwords, lemmatize, tokenize)
    
    Args: 
        document (str): The text to preprocess
        keep_negations (bool): whether to preserve negation words
        min_token_length(int): minimum token length to keep

    Returns: 
    '''

    # handle empty documents or non-strings (just incase)
    if not isinstance(document, str):
        return ''

    # handle some of the encoding issues
    document = document.replace('\x96', '—') # em dash
    document = document.replace('\x91', "'")  # Left single quote
    document = document.replace('\x92', "'")  # Right single quote
    document = document.replace('\x93', '"')  # Left double quote
    document = document.replace('\x94', '"')  # Right double quote
    document = document.replace('\x97', '—')  # Em dash
    
    # Handle contractions before removing punctuation
    if keep_negations:
        contractions = {
            r"n't": " not",
            r"'m": " am",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would"
        }
        for pattern, replacement in contractions.items():
            document = re.sub(pattern, replacement, document)


    # remove any html or url entities
    document = html_pattern.sub('', document)
    document = url_pattern.sub('', document)
    # strip any leading or trailing whitespace and lowercase
    document = document.strip().lower()
    document = punctuation_pattern.sub(' ', document) # remove punctuation using regular expressions (escaped punctuation string)
     
    tokens = word_tokenize(document) # tokenize the cleaned document using nltk method so we can easily remove stop words / lemmatize
    
    # Filter: remove stopwords, short tokens(at least length of <min_token_length>), and lemmatize
    tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stopw and len(word) >= min_token_length
    ]

    cleaned_doc = ' '.join(tokens) # join tokens back together, separated by a space

    return cleaned_doc

