from collections import Counter
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import numpy as np
from numpy.lib.function_base import average
from scipy.sparse import lil_matrix, csr_matrix
from irs import InformationRetrievalSystem

class VectModelInformationRetrievalSystem(InformationRetrievalSystem):
    def __init__(self, alpha, dataset):

        self.alpha = alpha  
        self.searched = {}
        
        if dataset == '1':  
            with open('datasets/Cranfield/CRAN.ALL.json') as data:    
                self.dataset = json.load(data)
            
            with open('datasets/Cranfield/CRAN.QRY.json') as data:    
                self.querys = json.load(data)

            with open('datasets/Cranfield/CRAN.REL.json') as data:    
                self.rel = json.load(data)
        elif dataset == '2':

            with open('datasets/Med/MED.ALL.json') as data:    
                self.dataset = json.load(data)  

            with open('datasets/Med/MED.QRY.json') as data:    
                self.querys = json.load(data)      
            
            with open('datasets/Med/MED.REL.json') as data:    
                self.rel = json.load(data)    
        else:
            raise Exception

        self.data = {}
        self.relevant_docs = int(average([len(queries.values()) for queries in self.rel.values()]))

        for doc in self.dataset.values():
            self.data[doc['id']] = {
                'id' : doc['id'],
                'title' : word_tokenize(str(self.__preprocess(doc['title']))) if 'title' in doc.keys() else [],
                'text' : word_tokenize(str(self.__preprocess(doc['abstract']))) if 'abstract' in doc.keys() else []
            }

        self.N = len(self.data)
        self.__df()
        self.__tf_idf()

        for query in self.querys.values():
            self.search(query['text'], query_id = query['id'])
    
    @staticmethod
    def __convert_lower_case(data):
        return np.char.lower(data)

    @staticmethod
    def __remove_stop_words(data):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    @staticmethod
    def __remove_punctuation(data):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        return data

    @staticmethod
    def __remove_apostrophe(data):
        return np.char.replace(data, "'", "")

    @staticmethod
    def __stemming(data):
        stemmer= PorterStemmer()
        
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text
    
    def __doc_freq(self, word):
        c = 0
        try:
            c = self.word_frequency[word]
        except:
            pass
        return c
    
    def __preprocess(self, data):
        data = VectModelInformationRetrievalSystem.__convert_lower_case(data)
        data = VectModelInformationRetrievalSystem.__remove_punctuation(data) #remove comma seperately
        data = VectModelInformationRetrievalSystem.__remove_apostrophe(data)
        data = VectModelInformationRetrievalSystem.__remove_stop_words(data)
        data = VectModelInformationRetrievalSystem.__stemming(data)
        data = VectModelInformationRetrievalSystem.__remove_punctuation(data)
        data = VectModelInformationRetrievalSystem.__stemming(data) #needed again as we need to stem the words
        data = VectModelInformationRetrievalSystem.__remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
        data = VectModelInformationRetrievalSystem.__remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
        return data
    
    def __df(self):
        self.word_frequency = {}

        for doc in self.data.values():
            for w in doc['title']:
                try:
                    self.word_frequency[w].add(int(doc['id']))
                except:
                    self.word_frequency[w] = {int(doc['id'])}

            for w in doc['text']:
                try:
                    self.word_frequency[w].add(int(doc['id']))
                except:
                    self.word_frequency[w] = {int(doc['id'])}

        for i in self.word_frequency:
            self.word_frequency[i] = len(self.word_frequency[i])
        
        self.total_vocab = [x for x in self.word_frequency]
        self.total_vocab_size = len(self.total_vocab)

    def __tf_idf(self):
        
        self.tf_idf = lil_matrix((self.N + 1, self.total_vocab_size))

        tf_idf = {}
        tf_idf_title = {}

        for doc in self.data.values():
                  
            counter = Counter(doc['text'])
            words_count = len(doc['text'])

            counter_title = Counter(doc['title'] + doc['text'])
            words_count_title = len(doc['title'] + doc['text'])
            
            for token in np.unique(doc['text']):
                
                tf = counter[token]/words_count
                df = self.__doc_freq(token)
                idf = np.log((self.N+1)/(df+1))
                
                tf_idf[int(doc['id']), token] = tf*idf

                tf_title = counter_title[token]/words_count_title
                df_title = self.__doc_freq(token)
                idf_title = np.log((self.N+1)/(df_title+1))
                
                tf_idf_title[int(doc['id']), token] = tf_title*idf_title

        for i in tf_idf:
            tf_idf[i] *= self.alpha
        
        for i in tf_idf_title:
            tf_idf[i] = tf_idf_title[i]

        
        for i in tf_idf:
            ind = self.total_vocab.index(i[1])
            self.tf_idf[i[0], ind] = tf_idf[i]
        
        self.tf_idf = csr_matrix(self.tf_idf)
        
    def __gen_query_vector(self, tokens, alpha):

        Q = lil_matrix((1, self.total_vocab_size))
        
        counter = Counter(tokens)
        words_count = len(tokens)
        
        for token in np.unique(tokens):
            
            tf = alpha + (1 - alpha) * (counter[token] / words_count)
            df = self.__doc_freq(token)
            if df:
                idf = math.log((self.N)/(df))
            else:
                idf = 0
            
            try:
                ind = self.total_vocab.index(token)
                Q[0, ind] = tf*idf
            except:
                pass
           
        return csr_matrix(Q)
    
    def __print_search(self, out, preview):
        for doc in out:
            print(f"{doc[0]} - { self.dataset[str(doc[0])]['title'] if self.dataset[str(doc[0])]['title'] != '' else 'Not Title'}\nText: {self.dataset[str(doc[0])]['abstract'][:preview]}")
            print()
            
    @staticmethod
    def __cosine_sim(a, b):
        return 0 if not a.max() or not b.max() else a.dot(b.transpose())/(VectModelInformationRetrievalSystem.__sparse_row_norm(a)*VectModelInformationRetrievalSystem.__sparse_row_norm(b))
    
    @staticmethod
    def __sparse_row_norm(A):
        out = np.zeros(A.shape[0])
        nz, = np.diff(A.indptr).nonzero()
        out[nz] = np.sqrt(np.add.reduceat(np.square(A.data),A.indptr[nz]))
        return out
    
    def search(self, query = '', alpha = 0.4, query_id = False, k = -1, preview = 500):
        if query_id and query_id in self.searched.keys():
            self.__print_search(self.searched[query_id][1][:self.relevant_docs], preview)
            return
        
        preprocessed_query = self.__preprocess(query)
            
        if (not query_id):
            print("\n---------- Ejecutando Búsqueda -----------\n")
        
        tokens = word_tokenize(str(preprocessed_query))
        d_cosines = []
        
        query_vector = self.__gen_query_vector(tokens, float(alpha))
        
        for d in self.tf_idf:
            d_cosines.append(VectModelInformationRetrievalSystem.__cosine_sim(d, query_vector))

        out = [(id, d_cosines[id].max()) for id in np.array(d_cosines, dtype = object).argsort()[-k:][::-1] if d_cosines[id] and d_cosines[id].max() > 0.0]

        if query_id:
            self.searched[query_id] = (query_vector, out)
        else:
            self.__print_search(out[:self.relevant_docs], preview)
