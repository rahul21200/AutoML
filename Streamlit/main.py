import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
import sys
# sys.addpath("../PyCaret")
# print("hello")
import matplotlib.pyplot as plt
from datapp import DataPreProcessing


from AutoClean import AutoClean
# import numpy as np

st.markdown(  # used for formating the app layout
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 100%;
        padding-top: 0rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 0rem;
    }}
</style>
""", unsafe_allow_html=True,)
st.title('Visualization')

def showCSV(uploaded_file):
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        print(dataframe)
        pipeline = AutoClean(dataframe,encode_categ=[False])
        st.write(pipeline.output)
        # arr = np.random.normal(0, 1, size=(100, 2))
        # fig, ax = plt.subplots()
        # ax.hist(arr, bins=20)

        # st.pyplot(fig)
        # dataframe.dropna(inplace=True)
        # df = sns.load_dataset('iris')
        # fig = plt.figure()
        # dataframe.drop("Species" , axis=1, inplace=True)
        # dataframe.drop("Id" , axis=1, inplace=True)
        # column_headers = list(df.columns.values)
        s = sns.pairplot(pipeline.output,hue="Sex")
        fig = s.fig
        st.pyplot(fig)
        # plt.show()
        # print(dataframe)
        # st.line_chart(data=dataframe)


uploaded_file = st.file_uploader("Choose a file")
# mod = DataPreProcessing(uploaded_file)
# # mod.data_stat()
# mod.setTarget('Sex')
# mod.remove_outliers()
# mod.feature_selection()
# mod.remove_multicollinearity()
# mod.pca()
# print(type(uploaded_file))
# dataframe = pd.read_csv(uploaded_file)
# pipeline = AutoClean()

# print(pipeline.output)

if st.button("View the uploaded data-frame"):
    showCSV(uploaded_file)
