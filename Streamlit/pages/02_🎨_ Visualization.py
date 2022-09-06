from autoviz.AutoViz_Class import AutoViz_Class
import os
import streamlit as st
    
AV = AutoViz_Class()


st.title('Visualize your data')


def generateGraphs():
    df = AV.AutoViz('temp.csv', verbose=2, chart_format='png')


generateGraphs()

for graphs in os.listdir('../AutoViz_plots/AutoViz'):
    if graphs.endswith(".png"):
        st.image('../AutoViz_plots/AutoViz/'+graphs, use_column_width=True)

