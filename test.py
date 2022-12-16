import streamlit as st
import pandas as pd

def make_visual_evaluation():

    metrics = {"presition": ['0.5','0.7','0'], "Recall":['0.8','0.7','0.9'], "F1":['0.7','0','0'], "Fallout":['0.9', '0', '0.7']}
 
  
  
    data = {'name': ['nick', 'david', 'joe', 'ross'],
        'age': ['5', '10', '7', '6']} 
        
    new = pd.DataFrame.from_dict(metrics)

    st.bar_chart(new)

if st.button("Show evaluation measures statistics"):
    make_visual_evaluation()