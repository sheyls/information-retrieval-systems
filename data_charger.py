import json
from pathlib import Path
from typing import Any, List, Dict
from os import mkdir

# Get the path of the actual directory 
PATH = Path.cwd()

# Charge amd structure the data of a data-set
class DataCharger:

    def __init__(self, dataset_name, dataset_path = f"{PATH}/data-sets"):

        self.data_name = dataset_name
        self.data_original_path = dataset_path
        self.data_path = f"{PATH}/data-processed/{dataset_name}"
        self._charge_data()

    def _charge_data(self):
        metadata: List[Dict[str, Any]] = []
        docs: List[str] = []
        mkdir(self.data_path)
        self._charge_metadata(metadata)
        self._charge_docs(docs)

    def _charge_metadata(self, metadata: List[dict]):

        metadata_path = self.data_path+"/metadata.json"
        with open(str(metadata_path), "w", encoding="utf-8") as file:
            json.dump(metadata, file)

    def _charge_docs(self, docs: List[str]):

        docs_path = self.data_path+"/docs.json"
        with open(str(docs_path), "w", encoding="utf-8") as file:
            json.dump(docs, file)

#TESTING
test_bd = DataCharger("cranfield")