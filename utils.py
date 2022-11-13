import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def read_json(dataset):
    if dataset == '1':  
            with open('datasets/Cranfield/CRAN.ALL.json') as data:    
                sol_dataset = json.load(data)
            
            with open('datasets/Cranfield/CRAN.QRY.json') as data:    
                sol_querys = json.load(data)

            with open('datasets/Cranfield/CRAN.REL.json') as data:    
                sol_rel = json.load(data)
    elif dataset == '2':

        with open('datasets/Med/MED.ALL.json') as data:    
            sol_dataset = json.load(data)  

        with open('datasets/Med/MED.QRY.json') as data:    
            sol_querys = json.load(data)      
        
        with open('datasets/Med/MED.REL.json') as data:    
            sol_rel = json.load(data)    
    else:
        raise Exception
    return sol_dataset, sol_querys, sol_rel