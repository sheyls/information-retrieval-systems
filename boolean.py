import numpy as np
from numpy.lib.function_base import average
from nltk.stem.snowball import EnglishStemmer
from irs import InformationRetrievalSystem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import utils
from nltk.tokenize import word_tokenize

class BooleanModel(InformationRetrievalSystem):
    def __init__(self,alpha, dataset) -> None:
        super().__init__()
        
        self.dataset, self.querys, self.rel = utils.read_json(dataset)
        self.data = {}
        self.relevant_docs = int(average([len(queries.values()) for queries in self.rel.values()]))

        for doc in self.dataset.values():
            self.data[doc['id']] = {
                'id' : doc['id'],
                'title' : word_tokenize(str(self.preprocess(doc['title']))) if 'title' in doc.keys() else [],

                'text' : word_tokenize(str(self.preprocess(doc['abstract']))) if 'abstract' in doc.keys() else []
            }
        
        self.corpus = self.GetCorpusRep()
        
        _stemmer = EnglishStemmer()
        _analyzer = CountVectorizer().build_analyzer()

        self.analyzer = self.__StemmerAnalyzer__(_stemmer, _analyzer)


        self.counter= CountVectorizer(analyzer= self.analyzer, binary= True, max_features=5000)
        X = self.counter.fit_transform(self.corpus)
        self.transformer = TfidfTransformer()
        self.transformer.fit(X)


    def __StemmerAnalyzer__(self, stemmer, analyzer):
        def stemmedWords(doc): return (stemmer.stem(w) for w in analyzer(doc))
        return stemmedWords

    def GetCorpusRep(self):
        corpus= [ ' '.join(doc['title']) + ' '.join(doc['text']) for doc in self.data.values()]
        return corpus
    


    def Transform_query(self,query: str):
        if isinstance(query, str):
            query = [query]
        

        result= np.array(self.counter.transform(query).indices)
        return result


    def __print_search(self, out, preview):
        for doc in out:
            print(f"{doc[0]} - { self.dataset[str(doc[0])]['title'] if self.dataset[str(doc[0])]['title'] != '' else 'Not Title'}\nText: {self.dataset[str(doc[0])]['abstract'][:preview]}")
            print()
    
    def search(self, query, alpha):
        processed_query= self.Transform_query(query)
        out= [doc for doc in self.corpus if not any(np.setdiff1d(processed_query, doc.tokens))]
        self.__print_search(out[:self.relevant_docs], preview=500)


""" corpus= ["Hola hola lindo hola mundo","feo"]
query ="hola lindo"
bm= BooleanModel()
print(bm.Transform_query(corpus,query)) """