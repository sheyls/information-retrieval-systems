
from virs import *
from typing import Dict, Tuple
import re
from pathlib import Path
from utils import read_json

def evaluate(corpus, model):

    precision_list = []
    recall_list = []
    f1_list = []
    fallout_list = []
    doc = {}
    queries = {}
    query_doc_relevance = {}
    total_docs = 0
    
    if corpus == "Cranfield":
        doc, queries, query_doc_relevance = utils.read_json("1")
        total_docs = len(doc)
        print(len(doc))

    elif corpus == "Med":
        doc, queries, query_doc_relevance = utils.read_json("2")
        total_docs = len(doc)

    else: print("Sorry, we don't have that corpus")

    for query in queries.values():

        r = model.search(query["text"])
       #cantidad de doc recuperado
        total_rel_doc, total_irrel_doc = get_total_relevant_documents(corpus, query,query_doc_relevance, total_docs)
        top = min(20,len(r))
        rr, ir = get_relevant_docs_count(corpus, r,query,query_doc_relevance,top) #number of relevant documents retrieved, number of relevant documents retrieved.
        try:
            precision = rr/top
        except:
            precision = 0
        try:
            recall = rr/top
        except:
            recall = 0
        print(rr)
        print(ir)
        precision_list.append(precision)
        try:
            recall = rr/total_rel_doc
        except:
            recall = 0
        recall_list.append(recall)
        f1 = calculate_measure_f(1,precision,recall)
        f1_list.append(f1)
        fallout = ir/total_irrel_doc
        fallout_list.append(fallout)
    [print(i) for i in precision_list]
    print("recobrado")
    [print(i) for i in recall_list]
    return precision_list, recall_list, f1_list, fallout_list

def calculate_measure_f(beta : float, precision : float, recall : float) -> float:

    if recall > 0 and precision > 0:
        return (1 + beta**2)/(1/precision)+(beta**2/recall)
    else:
        return 0

def get_total_relevant_documents(name, query, query_doc_relevance, total_docs_count : int ) -> int:
 
    relevant = 0
    for item in query_doc_relevance.keys():
        if query["id"] == item:
            for doc_rel in query_doc_relevance[item]:
                print(item)
                print(doc_rel)
                if(name == "Cranfield"):
                    if int(query_doc_relevance[item][doc_rel]["ground_truth"]) >= 2:
                        print("INTO")
                        print(int(query_doc_relevance[item][doc_rel]["ground_truth"]))
                        relevant+=1
                if(name == "Med"):
                    if int(query_doc_relevance[item][doc_rel]["ground_truth"]) >= 1:
                        print("INTO")
                        print(int(query_doc_relevance[item][doc_rel]["ground_truth"]))
                        relevant+=1
    print(f"RELEVANT:{relevant}")
    irrelevant = total_docs_count - relevant
        
    return relevant, irrelevant
            
def get_relevant_docs_count(name, r , query, query_doc_relevance: Dict[Tuple[int,int], int], top:int) -> int:

    rr = 0
    ir = 0
    

    for i in range(top):
        doc = r[i]
        try:
            rel = int(query_doc_relevance[query["id"]][doc["id"]]["ground_truth"])
            if name == "Cranfield":
                if rel > 2:
                    rr +=1
                else:
                    ir+=1
            if name == "Med":
                if rel >= 1:
                    rr +=1
                else:
                    ir+=1
            
        except:
            ir+=1
    
    return rr, ir

"""
d, q, r = utils.read_json('2')
v = VectorialModel(0.5, d, q, r )
evaluate("Med", v)"""