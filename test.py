import streamlit as st
import pandas as pd

def make_visual_evaluation():

    chart_data = pd.DataFrame(
    [
        [1, "metric1"],
        [2, "metric2"],
        [3, "metric3"],
    ],
    columns=["mean value", "metrics"])

    st.bar_chart(chart_data, x="metric name")

if st.button("Show evaluation measures statistics"):
    make_visual_evaluation()