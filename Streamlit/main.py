import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
# print("hello")
import matplotlib.pyplot as plt
import numpy as np


def showCSV(uploaded_file):
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        st.line_chart(data=dataframe, width=0, height=0,
                      use_container_width=True)


uploaded_file = st.file_uploader("Choose a file")
if st.button("View the uploaded data-frame"):
    showCSV(uploaded_file)
