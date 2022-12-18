from statistics import mean
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from virs import VectorialModel
from boolean import BooleanModel
from eval import evaluate


DOCS = None
RETRO = False
QUERY = ""
DATASET = None
ALPHA = 0.5
RETRO = False
MODEL = None
M_INC = None

def reset_search():
    st.session_state.results = []
    


def reset():
    DOCS = None
    QUERY = ""
    DATASET = None
    ALPHA = 0.5
    RETRO = False
    MODEL = None
    M_INC = None
    reset_search() 


def reset_search():
    st.session_state.results = []

def show_result(result: dict):
    doc_id = result["id"]
    expander_header = f"{doc_id} -- ".format(doc_id)
    if result["title"] != '':
        expander_header += result["title"].capitalize()
        
    with st.expander(f"{expander_header}"):
        if result["title"] != '' :
            t = result["title"]
            a = result["author"]
            st.caption(f"**{t.upper()}:** {a}")
    
        st.markdown(result["abstract"].capitalize())

st.title("Information Retrieval Multi-Models")
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
    DATASET = dataset


with col2:
    model = st.selectbox(
        "Choose a retrieval model", ["-", "Vectorial", "Boolean"])
    MODEL = model
    if not MODEL:
        st.error("Please select one model.")
    if dataset != "-":
        if MODEL == "Vectorial":
                irsystem = VectorialModel(ALPHA, dic[dataset])
                M_INC = irsystem
        elif MODEL == "Boolean":
                irsystem = BooleanModel(ALPHA, dic[dataset])
                M_INC = irsystem
    else: print("NO model")



coll1, coll2 = st.columns([6, 2])

if MODEL == "Vectorial":
    with coll2:
        RETRO = st.checkbox("Rocchio retroalimentation")
        print(RETRO)

    with coll1:
        query = st.text_input("Enter a query", placeholder="Write your query here")
        QUERY = query
        if  M_INC != None:
            if RETRO:
                result = M_INC.search(query, ALPHA)
                st.write(f"Found {len(result)} results")
                for r in result:
                    show_result(r)
                relevant = st.text_input("Write the document's ID you found relevants")
                M_INC.executeRocchio(QUERY, relevant, 1, 0.9, 0.5)
                M_INC.search(QUERY, preview=250)

            else:
                result = M_INC.search(query, ALPHA)
                st.write(f"Found {len(result)} results")
                for r in result:
                    show_result(r)

        else:
            print("No model instance")
             
    alpha = st.slider(
            "Smoothing constant",
            0.0,
            1.0,
            0.5,
            help="Smoothing constant",
        )
    ALPHA = alpha

elif model == "Boolean":
    query = st.text_input("Enter a query", placeholder="Write your query here")
    QUERY = query
    if M_INC != None:
        result = M_INC.search(query, ALPHA)
        st.write(f"Found {len(result)} results")
        for r in result:
            show_result(r)

else: print("No model selected")


def make_visual_evaluation():

    ps, rs, f1, fou = evaluate(DATASET, M_INC)
    p = mean(ps)
    r = mean(rs)
    f = mean(f1)
    o = mean(fou)
    data = pd.DataFrame([[p, "Presition"],[r, 'Recall'], [f, 'F1'], [o, 'Fallout']], columns=['Mean Value', 'Metrics'])
    st.bar_chart(data, x="Metrics")

if st.button("Show evaluation measures statistics"):
    make_visual_evaluation()