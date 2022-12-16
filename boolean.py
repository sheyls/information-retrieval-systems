import numpy as np
from numpy.lib.function_base import average
from nltk.stem.snowball import EnglishStemmer
from irs import InformationRetrievalSystem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import utils
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords

from matplotlib import pyplot as plot
import pandas as pd

from fbquery import is_binaryoperator
from fbquery import convert
from fbquery import is_Rparanthesis
from fbquery import is_Lparanthesis

from collections import defaultdict

import time

from edit_distance import minEditDistance

class BooleanModel(InformationRetrievalSystem):
    def __init__(self,alpha, dataset) -> None:
        super().__init__()
        
        self.searched = {}
        self.dataset, self.querys, self.rel = utils.read_json(dataset)
        self.data = {}
        self.relevant_docs = int(average([len(queries.values()) for queries in self.rel.values()]))

        for doc in self.dataset.values():
            self.data[int(doc['id'])] = {
                'id' : doc['id'],
                'title' : word_tokenize(str(self.preprocess(doc['title']))) if 'title' in doc.keys() else [],

                'text' : str(self.preprocess(doc['abstract'])) if 'abstract' in doc.keys() else ""
                #word_tokenize(str(self.preprocess(doc['abstract']))) if 'abstract' in doc.keys() else []
            }
        
        self.corpus = self.GetCorpusRep()

        #Stopwords
        self.stopword = set(stopwords.words("english"))
        
        #Stemming
        self.reverse_stem = defaultdict(list)
        self.ps = PorterStemmer()


        # Set of all unique terms in the corpus. We use a set to retrieve unique terms.
        self.dictionary = set()

        # Documents is a dictionary from key:documents to value:document_name
        self.doc = dict()

        # A posting list is a list of document identifiers (or document IDs) containing the term.
        self.postings = defaultdict(list)

        #Preprocessing: Producing a list of normalized token.
        self.corpus_preprocess()


    def GetCorpusRep(self):
        ''' Representation of the corpus as a list of strings. 
        Each string represents a document and has the structure:
        document_title + document_text'''
        corpus= [ ' '.join(doc['title']) + ' '.join(doc['text']) for doc in self.data.values()]
        return corpus
    
    def corpus_preprocess(self):
        start_time = time.time()
        """Preprocess the corpus"""

        """ Iterate through the list of documents in the folder to find
        all the unique words present after deleting numbers and
        special characters. Ignore the stopwords while finding the
        unique words. """
        i=1

        for item in self.data.values():
            text= item['text']
            # Removes all the punctutation marks/special characters from the text
            text= utils.remove_punctuation(text)
            # Tokenize text into words
            words = word_tokenize(text)
            # Remove stopwords
            # convert remaining words to lowercase
            words = [word.lower() for word in words if word not in self.stopword]
            #Stemming
            for word in words:
                self.reverse_stem[self.ps.stem(word)].append(word)
            for key in self.reverse_stem.keys():
                self.reverse_stem[key] = self.unique(self.reverse_stem[key])
            words = [self.ps.stem(word) for word in words]

            terms = self.unique(words)

            # Add posting to Final Posting List
            for term in terms:
                self.postings[term].append(i)
            # Make a list of indexed documents
            self.doc[i] = item['title']
            i=i+1

        # Making inverted index out of final posting list.
        self.dictionary = self.postings.keys()
        end_time = time.time()
        total_time = end_time - start_time
        print("Preprocessing + Indexing Time: ", total_time)
        return start_time

    def unique(self, words):
        """ Function to find all the unique words in the document
        passed as a parameter.Set ensures we get unique elements. 
        We then typecast a set to a list to return a list of unique words """ 
        return list(set(words))


    def query(self, query):
        start_time = time.time()
        """Query the indexed documents using a boolean model
        :query: valid boolean expression to search for
        :returns: list of matching document names
        """

        
        # Tokenize query
        q = word_tokenize(query)
        
        # Convert infix query to postfix query
        q = convert(q)
        
        
        # Evaluate query against already processed documents
        docs = self.search(q)
        
        end_time = time.time()
        total_time = end_time - start_time
        print("Searching Time: ", ("{0:.14f}".format(total_time)))
        self.__print_search(docs, 500)


    def __print_search(self, out, preview):
        print(out)
        for doc in out:
            print(f"{doc[0]} - { self.dataset[str(doc[0])]['title'] if self.dataset[str(doc[0])]['title'] != '' else 'Not Title'}\nText: {self.dataset[str(doc[0])]['abstract'][:preview]}")
            print()
    
    def search(self, query, query_id = False, alpha=0.5):
        """Evaluates the query
        returns names of matching document 
        """
        if query_id and query_id in self.searched.keys():
            self.__print_search(self.searched[query_id][1][:self.relevant_docs], preview=500)
            return

        word = []
        # query: list of query tokens in postfix form
        for token in query:
            searched_token = token
            # Token is an operator,
            # Pop two elements from stack and apply it.
            if is_binaryoperator(token):
                # Pop right operand
                if len(word)==0:
                    raise ValueError("Query is not correctly formed!")
                right_word = word.pop()

                # Pop left operand
                if len(word)==0:
                    raise ValueError("Query is not correctly formed!")
                left_word = word.pop()

                # Perform task
                doc_list = self.solve(left_word, right_word, token)

                word.append(doc_list)

            # Token is an operand, push it to the word
            else:        
                # Lowercasing and stemming query term
                token = self.preprocess(token)
                #token = self.ps.stem(token.lower())

                """ if token[0] == "~":
                    token = token[1:]"""
                    
                #token= self.lm.lemmatize(token.lower()) 
                
                # Edit distance
                threshold =2
                keys = []
                if token[0] == "~":
                    token = token[1:]
                
                """ Performing a spell check and correction on all the query
                words if the spelling is wrong. This is done by comparing
                the edit distance between the query words with all the
                unique words across all the documents. The word is then
                replaced by the word which has the minimum edit distance and 
                among those, the one having the largest posting list.
                If the query word exists in the documents, the minimum edit
                distance is zero and the word remains unchanged. """
                count=0
                for key in self.dictionary:
                    distance= minEditDistance(key,token,len(key),len(token))
                    if distance <= threshold:
                        count=count+1
                        for term in self.reverse_stem[key]:
                            if(threshold >= minEditDistance(term,token,len(term),len(token))):
                                keys.append(term)
                if count == 0:            
                    print( token," is not found in the corpus!" )
                    return np.zeros(len(self.data), dtype=bool)
                
                word.append(self.bits(searched_token,token,keys))
        
        if len(word) != 1:
            print("Wrong query!")
            return list()

        # Find out id of documents corresponding to set bits in the list
        docs = [i+1 for i in np.where(word[-1])[0]]
        
        return docs
    
    def __print_search(self, out, preview=500):
        for doc_id in out:
            print(f"{doc_id} - { self.dataset[str(doc_id)]['title'] if self.dataset[str(doc_id)]['title'] != '' else 'Not Title'}\nText: {self.dataset[str(doc_id)]['abstract'][:preview]}")
            print()
    
    def solve(self, left_word, right_word, b):
        """
        :b: binary operation to perform
         The final result is calculated and returned. 
         This list is then iterated through
         and whenever the value is true, it implies that the document
         satisfies the given boolean query and its name is displayed.
         If the value is false, it skips to the value of the next document
         in the resultant list. "
        """
        if b == "&":
            #searching for documents with both left and right query words
            return left_word & right_word
        elif b == "|":
            #searching for documents containing either left or right or both query words
            return left_word | right_word
        else:
            return 0

    def bits(self, token, word, other_words):
        """Make bit list out of a word
        :returns: bit list of word with bits set when it appears in the particular documents
        """
        # Size of bit list
        totalDoc = len(self.corpus)
        
        negation = False
        if token[0] == "~":  
            # unary not operator(~)
            negation = True

        if word in self.dictionary:
            
            # Intialize a binary list for the word
            binary_list = np.zeros(totalDoc, dtype=bool)

            # Locate query token in the dictionary and retrieve its posting list
            posting = self.postings[word]

            # Set bit=True for doc_id in which query token is present
            for doc_id in posting:
                binary_list[doc_id-1] = True

            if negation:
                # Instance of NOT operator token,
                # bit list is supposed to be negated
                """ Applies the unary not operator on the relevant query term.
                This is used to invert the values present in the list for that term."""
                length = len(binary_list)
                for i in range(length):
                    if binary_list[i] == True :
                        binary_list[i] = False
                    else:
                        binary_list[i] = True
                
            # Return bit list of the word
            return binary_list

        else:
            # Word is not present in the corpus
            if token[0]=="~":
                print(token[1:] ," is not found in the corpus!" )
            else:
                print(token," is not found in the corpus!")
            
            
            if(len(other_words)):
                print("Did you mean these ? : ")
            other_words = list((set(other_words) - set(self.dictionary)))
            #Printing all the words under the threshold value for mispelt tokens
            maxLength = 0
            min_word = ""
            if token[0]=="~":
                token=token[1:]
            for key in other_words:
                if maxLength < len(self.postings[self.ps.stem(key.lower())]):
                    maxLength = len(self.postings[self.ps.stem(key.lower())])
                    min_word = key
                elif maxLength == len(self.postings[self.ps.stem(key.lower())]):
                    if minEditDistance(key,token,len(key),len(token)) < minEditDistance(min_word,token,len(min_word),len(token)):
                        min_word = key
                     
            print("Giving results based on: ",min_word)
            binary_list = np.zeros(totalDoc, dtype=bool)
            posting = self.postings[self.ps.stem(min_word)]
            for doc_id in posting:
                binary_list[doc_id-1] = True
            if negation:
                length = len(binary_list)
                for i in range(length):
                    if binary_list[i] == True :
                        binary_list[i] = False
                    else:
                        binary_list[i] = True
            return binary_list

    def evaluate_query(self, query_id, show_output):
        if str(query_id) not in self.searched.keys():
            print("Consulta no encontrada")
            return

        if (show_output):
            print("\nConsulta: " + self.queries[str(query_id)]['text']) 

        return self.__evaluate(self.searched[query_id][1],self.rel[str(query_id)], show_output)

    def __evaluate(self, ranking, relevants_docs_query, show_output):
        
        [true_positives, false_positives] = self.__relevant_doc_retrieved(ranking, relevants_docs_query)

        recall = BooleanModel.__get_recall(true_positives,len(relevants_docs_query))
        precision = BooleanModel.__get_precision(true_positives,false_positives)
        if precision and recall:
            f1 = 2 / (1/precision + 1/recall)
        else:
            f1 = 0

        if show_output:
            print(f"\nPrecisión: {precision} \nRecobrado: {recall} \nMedida F1: {f1}\n")

            true_positives = 0
            false_positives = 0
            recall = []
            precision = []
            for doc in ranking:
                if str(doc[0]) in relevants_docs_query.keys():
                    true_positives += 1
                else:
                    false_positives += 1

                recall.append(self.__get_recall(true_positives,len(relevants_docs_query)))
                precision.append(self.__get_precision(true_positives,false_positives))


            recalls_levels = np.array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]) 

            interpolated_precisions = self.__interpolate_precisions(recall, precision, recalls_levels)
            self.__plot_results(recalls_levels, interpolated_precisions)
            return
        else:
            return recall, precision, f1
    
    def __relevant_doc_retrieved(self, ranking, relevants_docs_query):
        true_positives = 0
        false_positives = 0
        for doc in ranking[:self.relevant_docs]:
           if str(doc[0]) in relevants_docs_query.keys():
                true_positives += 1
           else:
                false_positives += 1
        return true_positives, false_positives
        
    @staticmethod
    def __get_recall(true_positives, real_true_positives):
        recall=float(true_positives)/float(real_true_positives)
        return recall
    
    @staticmethod
    def __get_precision(true_positives, false_positives):
        relevant_items_retrieved=true_positives+false_positives
        precision=float(true_positives)/float(relevant_items_retrieved)
        return precision
    
    @staticmethod
    def __interpolate_precisions(recalls,precisions, recalls_levels):
        precisions_interpolated = np.zeros((len(recalls), len(recalls_levels)))
        i = 0
        while i < len(precisions):
            # use the max precision obtained for the topic for any actual recall level greater than or equal the recall_levels
            recalls_inter = np.where((recalls[i] > recalls_levels) == True)[0]
            for recall_id in recalls_inter:
                if precisions[i] > precisions_interpolated[i, recall_id]:
                    precisions_interpolated[i, recall_id] = precisions[i]
            i += 1

        mean_interpolated_precisions = np.mean(precisions_interpolated, axis=0)
        return mean_interpolated_precisions

    @staticmethod
    def __plot_results(recall, precision):
        plot.plot(recall, precision)
        plot.xlabel('Recobrado')
        plot.ylabel('Precisión')
        plot.draw()
        plot.title('P/R')
        plot.show()
        
    def evaluate_system(self):
        print("\n---------- Ejecutando Evaluación General del Sistema -----------\n")
        sum_recall = 0
        sum_precision = 0
        sum_f1 = 0
        sum_errors = 0
        for query in self.searched.keys():
            recall, precision, f1 = self.evaluate_query(query, False)
            sum_recall += recall
            sum_precision += precision
            sum_f1 += f1
            if not f1:
                sum_errors += 1
        
        print(f'Promedio de Precisión: {sum_precision/len(self.querys)} \nPromedio de Recobrado: {sum_recall/len(self.querys)} \nPromedio de Medida F1: {sum_f1/len(self.querys)} \nNingún Documento Relevante Recuperado: {sum_errors} Veces ({sum_errors*100/len(self.querys)}%)\n')
       

corpus= ["Hola hola lindo hola mundo","feo"]
queryy ="avion"
ask = [f'{query[0]} - {bm.querys[query[0]]["text"]}\n' for query in bm.searched.items()]
bm= BooleanModel(0.5, "2")
bm.query(ask)
query = input("".join(ask) + 'Elegir ID -> ')
bm.evaluate_query(query, True)