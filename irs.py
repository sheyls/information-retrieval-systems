from abc import ABC, abstractmethod
import utils


class InformationRetrievalSystem(ABC):
    @abstractmethod
    def search(self, query):
        pass
    def preprocess(self, data):
        data = utils.convert_lower_case(data)
        data = utils.remove_punctuation(data) #remove comma seperately
        data = utils.remove_apostrophe(data)
        data = utils.remove_stop_words(data)
        data = utils.stemming(data)
        #data = utils.remove_punctuation(data)
        data = utils.stemming(data) #needed again as we need to stem the words
        data = utils.remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
        data = utils.remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
        return data
    def evaluate_system(self):
        pass