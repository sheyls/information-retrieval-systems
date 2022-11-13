from abc import ABC, abstractmethod


class InformationRetrievalSystem(ABC):
    @abstractmethod
    def search(self, query):
        pass