from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np
import math
from numpy.lib.function_base import average
from scipy.sparse import lil_matrix, csr_matrix
from irs import InformationRetrievalSystem
import utils

class VectorialModel(InformationRetrievalSystem):
    def __init__(self, alpha, dataset):

        self.alpha = alpha  
        self.searched = {}
        
        self.dataset, self.queries, self.rel = utils.read_json(dataset)

        self.data = {}
        self.relevant_docs = int(average([len(queries.values()) for queries in self.rel.values()]))

        for doc in self.dataset.values():
            self.data[doc['id']] = {
                'id' : doc['id'],
                'title' : word_tokenize(str(self.preprocess(doc['title']))) if 'title' in doc.keys() else [],
                'text' : word_tokenize(str(self.preprocess(doc['abstract']))) if 'abstract' in doc.keys() else []
            }

        self.N = len(self.data)
        self.__df()
        self.__tf_idf()
    
    
    def __doc_freq(self, word):
        c = 0
        try:
            c = self.word_frequency[word]
        except:
            pass
        return c
    
    
    
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
        resul = []
        for doc in out:
            resul.append(self.dataset[str(doc[0])])
            print(f"{doc[0]} - { self.dataset[str(doc[0])]['title'] if self.dataset[str(doc[0])]['title'] != '' else 'Not Title'}\nText: {self.dataset[str(doc[0])]['abstract'][:preview]}")
        return resul
            
    @staticmethod
    def __cosine_sim(a, b):
        return 0 if not a.max() or not b.max() else a.dot(b.transpose())/(VectorialModel.__sparse_row_norm(a)*VectorialModel.__sparse_row_norm(b))
    
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
        
        preprocessed_query = self.preprocess(query)
            
        if (not query_id):
            print("\n---------- Ejecutando Búsqueda -----------\n")
        
        tokens = word_tokenize(str(preprocessed_query))
        d_cosines = []
        
        query_vector = self.__gen_query_vector(tokens, float(alpha))
        
        for d in self.tf_idf:
            d_cosines.append(VectorialModel.__cosine_sim(d, query_vector))

        out = [(id, d_cosines[id].max()) for id in np.array(d_cosines, dtype = object).argsort()[-k:][::-1] if d_cosines[id] and d_cosines[id].max() > 0.0]

        if query_id:
            self.searched[query_id] = (query_vector, out)
        else:
            return self.__print_search(out[:self.relevant_docs], preview)
            
    def executeRocchio(self, query_id, relevants, alpha, beta, gamma):
        if query_id in self.searched.keys():
            query = self.searched[query_id]

            query_vector = query[0]

            rel_docs = 0
            sum_rel_docs = lil_matrix((1, self.total_vocab_size))
            nonrel_docs = 0
            sum_nonrel_docs = lil_matrix((1, self.total_vocab_size))

            for doc in query[1][:self.relevant_docs]:
                if str(doc[0]) in relevants:
                    rel_docs += 1
                    sum_rel_docs += self.tf_idf[doc[0]]
                else:
                    nonrel_docs += 1
                    sum_nonrel_docs += self.tf_idf[doc[0]]
                
            term1 = [alpha*word for word in query_vector.toarray()]

            sum_rel_docs = sum_rel_docs.toarray()
            sum_nonrel_docs = sum_nonrel_docs.toarray()

            pos=0   
            while pos < len(term1[0]):
                term1[0][pos] += (float(beta)/rel_docs) * sum_rel_docs[0][pos] - (float(gamma)/nonrel_docs) * sum_nonrel_docs[0][pos]
                pos += 1  
            
            d_cosines = []
            for d in self.tf_idf:
                d_cosines.append(VectorialModel.__cosine_sim(d, csr_matrix(term1[0])))

            out = [(id, d_cosines[id].max()) for id in np.array(d_cosines).argsort()[1:][::-1] if d_cosines[id] and d_cosines[id].max() != 0.0]
            
            self.searched[query_id] = (csr_matrix(term1), out)