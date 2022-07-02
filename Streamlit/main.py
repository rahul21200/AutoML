# from autoviz.AutoViz_Class import AutoViz_Class
from dataclasses import dataclass
from click import confirm
import streamlit as st
import os
import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from AutoClean import AutoClean
from datetime import datetime
import numpy as np


st.set_page_config(layout="wide", page_title="Visualization",
                   initial_sidebar_state="expanded")

uploaded_file = st.file_uploader("Choose a file", key='file_upload')

if uploaded_file != None:
    dataframe = pd.read_csv(uploaded_file)
    if st.button("View the uploaded data-frame"):
        pipeline = AutoClean(dataframe, encode_categ=[False])
        st.write(pipeline.output)
    dataframe.to_csv('temp.csv')
    