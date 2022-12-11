from urllib.error import URLError
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as  plt
import numpy as np
from virs import VectModelInformationRetrievalSystem

def reset_search():
    st.session_state.results = []


def reset():
    st.session_state.docs = None
    st.session_state.model = None
    st.session_state.dataset = None
    st.session_state.smoothing = 0.5
    st.session_state.top = 10
    st.session_state.retro = False
    st.session_state.irsystem = None
    reset_search() 

state_vars = [
    "docs",
    "results",
    "model",
    "dataset",
    "from_i",
    "to_i",
    "query",
    "score",
    "top",
    "retro"
    "smoothing"
    "irsystem"
]
for var in state_vars:
    if var not in st.session_state:
        st.session_state[var] = None

def reset_search():
    st.session_state.results = []

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
    if st.session_state.docs is None or st.session_state.dataset != dataset:
        reset()
        st.session_state.dataset = dataset


with col2:
    model = st.selectbox(
        "Choose a retrieval model", ["-", "Vectorial", "Boolean", "PMA"])
    if not model:
        st.error("Please select one model.")
    if model == "Vectorial":
        if dataset != "-":
            irsystem = VectModelInformationRetrievalSystem(0.3, dic[dataset])
            st.session_state.irsystem = irsystem
    elif model == "Boolean":
        st.button("kk")



coll1, coll2 = st.columns([6, 2])

if model == "Vectorial":
    with coll2:
        st.session_state.retro = st.checkbox("Rocchio retroalimentation")
        print(st.session_state.retro)

    with coll1:
        query = st.text_input("Enter a query", placeholder="Write your query here")
        st.session_state.query = query
        if  st.session_state.retro:
            pass
        else:
            if st.session_state.irsystem != None:
                st.session_state.irsystem.search(query, st.session_state.smoothing)
elif model == "Boolean":
    with coll1:
        query = st.text_input("Enter a query", placeholder="Write your query here")
        st.session_state.query = query


cols = st.columns(2)

with cols[0]:
    if model == "Vectorial":
        smoothing = st.slider(
            "Smoothing constant",
            0.0,
            1.0,
            0.5,
            help="Smoothing constant",
        )
        st.session_state.smoothing = smoothing
    
     
#with cols[1]:
   # limit = st.slider(
   #     "Top", 1, len(st.session_state.docs), 10, help="Max number of results to show"
   # )
    #if st.session_state.top != limit:
    #    reset_search()
    #    st.session_state.top = limit


def make_visual_evaluation():
    ps, rs, fs, f1s, fls = [], [], [], [], []

    chart_data = pd.DataFrame(
        np.random.randn(20, 4),
        columns=['Precision', 'Recall','F metric', 'F1 metric'])

    st.line_chart(chart_data)

if st.button("Show evaluation measures statistics"):
    make_visual_evaluation()