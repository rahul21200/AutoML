from autoviz.AutoViz_Class import AutoViz_Class
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
# st.sidebar.title("Settings")

# Visualization part

AV = AutoViz_Class()


# @st.cache(allow_output_mutation=True)
# def showCSV(dataframe):
# s = sns.pairplot(pipeline.output)
# fig = s.fig
# return pipeline ,fig


# File Uploader
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file != None:
    dataframe = pd.read_csv(uploaded_file)
    dataframe.to_csv('temp.csv')

    @st.cache
    def generateGraphs(dataframe):
        df = AV.AutoViz('temp.csv', verbose=2, chart_format='png')
    generateGraphs(dataframe)

    if st.button("View the uploaded data-frame"):
        pipeline = AutoClean(dataframe, encode_categ=[False])
        # pipeline ,fig = showCSV(dataframe)
        st.write(pipeline.output)
    viz = st.expander("Visualization")
    for graphs in os.listdir('AutoViz_plots/AutoViz'):
        if graphs.endswith(".png"):
            viz.image('AutoViz_plots/AutoViz/'+graphs, use_column_width=True)

    option = st.radio('What is your ML model type?',
                      ('Classification', 'Regression'), index=0, horizontal=True)

    if option == "Classification":
        from pycaret.classification import *
        from classification import DataPreProcessing
        from pycaret.classification import *
    if option == "Regression":
        from pycaret.regression import *
        from regression import DataPreProcessing
        from pycaret.regression import *
    # with st.form("my_form"):
    st.header("Target")
    # print(list(file_)[0].split(','))
    # print(tuple(list(file_)[0].split(',')))
    target_ = st.selectbox("choose your target value",
                           tuple(list(dataframe)[::-1]), index=0)

    st.title("Data Preprocessing")
    # 'outliers_threshold': 0.05,
    # 'normalize_method': 'zscore',
    # 'feature_selection_threshold': 0.8,
    # 'multicollinearity_threshold': 0.9,
    # 'pca_components': 0.99,
    col1, col2 = st.columns(2)
    v1 = col1.checkbox("Remove Outliers", key='rem_outlier')
    if v1:
        v11 = col2.slider("Outlier Threshold", min_value=0.0,
                          max_value=0.3, value=0.05, step=0.01)
    v2 = col1.checkbox("Fix Imbalance", key='2')
    v3 = col1.checkbox("Normalize", key='3')
    if v3:
        v33 = col2.selectbox("Normalize Method", ("zscore",
                                                  "minmax", "maxabs", "robust",), index=0)
    v4 = col1.checkbox("Feature Selection", key='4')
    if v4:
        v44 = col2.slider("Feature Selection Threshold", min_value=0.0,
                          max_value=1.0, value=0.8, step=0.1)
    v5 = col1.checkbox("Remove Muticolinearity", key='5')
    if v5:
        v55 = col2.slider("Multicolinearity Threshold", min_value=0.0,
                          max_value=1.0, value=0.9, step=0.1)
    v6 = col1.checkbox("PCA", key='6')
    if v6:
        v66 = col2.slider("PCA components", min_value=0.0,
                          max_value=1.0, value=0.99, step=0.01)

    # submitted = st.form_submit_button("Submit")
    # if submitted:
    submit = st.button("Submit", key="submit")
    if submit:
        mod = DataPreProcessing(dataframe)
        if v1:
            mod.remove_outliers()
            # mod.outliers_threshold(v11)
        if v2:
            mod.fix_imbalance()
        if v3:
            mod.normalize()
            # mod.normalize_method(v33)
        if v4:
            mod.feature_selection()
            # mod.feature_selection_threshold(v44)
        if v5:
            mod.remove_multicollinearity()
            # mod.multicollinearity_threshold(v55)
        if v6:
            mod.pca()
            # mod.pca_components(v66)

        # print(mod.df, "-----", type(mod.df))
        mod.setTarget(target_.strip())
        models_ = mod.allSetup()
        # st.write(models_)
        print(models_[0])
        best_model = compare_models(verbose=False)
        model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_model(best_model, model_name)
        pull1 = pull()

        st.write(pull1)
        # lr = get_leaderboard()
        # pull2 = pull()
        # st.write(pull2)
        # st.write(type(best_model))
        # st.write(pd.DataFrame(models_[1]))
        # st.write(pd.DataFrame(models()))
        st.write("BEST MODEL")
        st.write(best_model)
        with open(model_name + ".pkl", "rb") as f:
            Confirm_download = st.download_button(
                "Download Model", f, file_name=f"{model_name}.pkl")
        if Confirm_download:
            os.remove(model_name + ".pkl")

        # os.remove("model.pkl")
        # st.download_button(label, data, file_name=None, mime=None, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)
        # st.download_button("Download Model", best_model, file_name="model.pkl")
        # print(type(best_model))
        # save_model(best_model, 'my_best_pipeline')
