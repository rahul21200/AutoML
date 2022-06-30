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
from pycaret.classification import *

from AutoClean import AutoClean
# import numpy as np

st.markdown(  # used for formating the app layout
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 200%;
        padding-top: 0rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 0rem;
    }}
</style>
""", unsafe_allow_html=True,)
st.title('Visualization')

dataframe = None
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()

df = AV.AutoViz('iris.csv',verbose=2,chart_format='png')

def showCSV(uploaded_file):
    # global dataframe
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        print(dataframe)
        pipeline = AutoClean(dataframe, encode_categ=[False])
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
        s = sns.pairplot(pipeline.output)
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

# print(type(dataframe))
# BUTTONS

# "remove_outliers": False,
# # "fix_imbalance": False,
# "normalize": False,
# "feature_selection": False,
# "remove_multicollinearity": False,
# "pca": False
# with st.form("my_form"):
with st.form("my_form"):
    v1 = st.checkbox("Remove Outliers", key='1')
    v2 = st.checkbox("Fix Imbalance", key='2')
    v3 = st.checkbox("Normalize", key='3')
    v4 = st.checkbox("Feature Selection", key='4')
    v5 = st.checkbox("Remove Muticolinearity", key='5')
    v6 = st.checkbox("PCA", key='6')
    submitted = st.form_submit_button("Submit")

    if submitted:
        file_ = pd.read_csv(uploaded_file)
        print(type(file_))
        mod = DataPreProcessing(file_)
        if v1:
            mod.remove_multicollinearity()
        if v2:
            mod.fix_imbalance()
        if v3:
            mod.normalize()
        if v4:
            mod.feature_selection()
        if v5:
            mod.remove_multicollinearity()
        if v6:
            mod.pca()
        mod.setTarget("Species")
# mod.allSetup()
        print(mod.df, "-----", type(mod.df))
        mod.allSetup()
        best_model = compare_models()
        st.write("BEST MODEL")
        st.write(best_model)
        print(type(best_model))
        save_model(best_model, 'my_best_pipeline')
