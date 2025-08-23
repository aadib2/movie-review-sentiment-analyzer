import nltk
import string
import re # regular expression

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# necessary packages to use imports above
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# define stopwords, punctuation to remove
stopw = set(stopwords.words('english')) # includes common words in book reviews
punctuation = re.escape(string.punctuation) # originally was using list but this is cleaner and faster, uses regular expressions

print(punctuation)

# define the lemmatizer from nltk (reduces words to their base form like acting-> act)
lemmatizer = WordNetLemmatizer()


# define preprocessing pipeline in method
def preprocess(document):
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
    
    # remove any html entities
    document = re.sub(r'<.*?>', '', document)
    # strip any leading or trailing whitespace and lowercase
    document = document.strip().lower()
    document = re.sub(rf'[{punctuation}]', '', document) # remove punctuation using regular expressions (escaped punctuation string)
     
    tokens = word_tokenize(document) # tokenize the cleaned document using nltk method so we can easily remove stop words / lemmatize
    tokens = [word for word in tokens if word not in stopw] # remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # lemmatize words
    
    cleaned_doc = ' '.join(tokens) # join tokens back together, separated by a space

    return cleaned_doc
