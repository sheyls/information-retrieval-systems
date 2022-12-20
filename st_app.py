from statistics import mean
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from virs import VectorialModel
from boolean import BooleanModel
from utils import read_json
from eval import evaluate

state_vars = [
    "docs",
    "docs_proc",
    "queries"
    "rel"
    "query",
    "results",
    "model",
    "dataset",
    "alpha",
    "retro",
    "vinstance",
    "binstance",
    "eval"
]
for var in state_vars:
    if var not in st.session_state:
        st.session_state[var] = None


def reset_search():
    st.session_state.results = []

def reset():
    st.session_state.docs = None
    st.session_state.queries = None
    st.session_state.rel = None
    st.session_state.model = None
    st.session_state.query = ""
    st.session_state.dataset = None
    st.session_state.alpha = 0.5
    st.session_state.alpha = False
    reset_search()


def show_result(result: dict):
    doc_id = result["id"]
    expander_header = f"{doc_id} - ".format(doc_id)
    if result["title"] != '':
        expander_header += result["title"].capitalize()
        
    with st.expander(f"{expander_header}"):
        if result["title"] != '' :
            t = result["title"]
            st.caption(f"**{t.upper()}:**")
    
        st.markdown(result["abstract"].capitalize())

st.title("Information Retrieval System")
reset()

dic = {"Cranfield": "1", "Med": "2"}

path = os.getcwd()

datasets = os.listdir(path+"/datasets")

if not datasets:
    st.write("No databases found")
datasets.insert(0, "-")

col1, col2 = st.columns([2, 2])

with col1:
    dataset = st.selectbox("Select a database", datasets)
    if dataset != "-":
        if st.session_state.docs is None or st.session_state.dataset != dataset:
            reset()
            docs, queries, rel = read_json(dic[dataset])

            st.session_state.docs = docs
            st.session_state.dataset = dataset
            st.session_state.queries = queries
            st.session_state.rel = rel


with col2:
    model = st.selectbox(
        "Choose a retrieval model", ["-", "Vectorial", "Boolean"])
    st.session_state.model = model
    if not model:
        st.error("Please select one model.")
    if dataset != "-":
        if model == "Vectorial":
            if st.session_state.vinstance == None:
                irsystem = VectorialModel(st.session_state.alpha, st.session_state.docs, st.session_state.queries, st.session_state.rel)
                st.session_state.vinstance = irsystem
                
        elif model == "Boolean":
            if st.session_state.binstance == None:
                irsystem = BooleanModel(st.session_state.docs, st.session_state.queries, st.session_state.rel)
                st.session_state.binstance = irsystem



coll1, coll2 = st.columns([6, 2])

if st.session_state.model == "Vectorial":
    with coll2:
        retro = st.checkbox("Rocchio retroalimentation")

    with coll1:
        query = st.text_input("Enter a query", placeholder="Write your query here")
        st.session_state.query = query
        if  st.session_state.vinstance != None and query != "":
            if retro:
                result = st.session_state.vinstance.search(query, st.session_state.alpha)
                st.write(f"Found {len(result)} results")
                for r in result:
                    show_result(r)
                relevant = st.text_input("Write the document's ID you found relevants")

                if relevant != "":
                    st.write(f"Do you find these more relevant?")
                    st.session_state.vinstance.executeRocchio(query, relevant, 1, 0.9, 0.5)
                    resul = st.session_state.vinstance.search(query, preview=250)
                    for r in result:
                        show_result(r)


            else:
                result = st.session_state.vinstance.search(query, st.session_state.alpha)
                st.write(f"Found {len(result)} results")
                for r in result:
                    show_result(r)

             
    alpha = st.slider(
            "Smoothing constant",
            0.0,
            1.0,
            0.5,
            help="Smoothing constant",
        )
    st.session_state.alpha = alpha

elif st.session_state.model == "Boolean":
    query = st.text_input("Enter a query", placeholder="Write your query here")
    st.session_state.query = query
    if query != "":
        result = st.session_state.binstance.search(query)
        st.write(f"Found {len(result)} results")
        for r in result:
            show_result(r)


def make_visual_evaluation():

    inc = 3
    if(st.session_state.model == "Vectorial"):
        inc = st.session_state.vinstance
    elif(st.session_state.model == "Booleano"):
        print("holaaa")   
        if inc != None:
            inc = st.session_state.binstance
        else:
            irsystem = BooleanModel(st.session_state.docs, st.session_state.queries, st.session_state.rel)
            st.session_state.binstance = irsystem
    else: 
        inc = st.session_state.binstance


    ps, rs, f1, fou = evaluate(st.session_state.dataset, inc)
    p = mean(ps)
    r = mean(rs)
    f = mean(f1)
    o = mean(fou)
    data = pd.DataFrame([[p, "Presition"],[r, 'Recall'], [f, 'F1'], [o, 'Fallout']], columns=['Mean Value', 'Metrics'])
    st.line_chart(data, x="Metrics")
    st.session_state.val = "simiherma"

if st.button("Show evaluation measures statistics"):
    make_visual_evaluation()

# what accurate or exact solutions of the laminar separation point for various incompressible