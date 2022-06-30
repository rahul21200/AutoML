import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from pycaret.classification import *
# import catboost
# import xgboost
import streamlit.components.v1 as components
from AutoClean import AutoClean
# import numpy as np
st.set_page_config(layout="wide")

# dataframe = None

# Visualization part


def showCSV(dataframe):
    if dataframe is not None:
        # Can be used wherever a "file-like" object is accepted:
        # print(dataframe)
        pipeline = AutoClean(dataframe, encode_categ=[False])
        st.write(pipeline.output)
        s = sns.pairplot(pipeline.output)
        fig = s.fig
        st.pyplot(fig)


# File Uploader
uploaded_file = st.file_uploader("Choose a file")


if uploaded_file != None:
    dataframe = pd.read_csv(uploaded_file)
    if st.button("View the uploaded data-frame"):
        showCSV(dataframe)
    # file_ = pd.read_csv(uploaded_file, delim_whitespace=True)

    # Classification or regression
    option = st.selectbox(
        'What is your ML model type?',
        ('Classification', 'Regression'), index=0)

    if option == "Classification":
        from classification import DataPreProcessing
    if option == "Regression":
        from regression import DataPreProcessing
    # with st.form("my_form"):
    st.title("Target")
    # print(list(file_)[0].split(','))
    # print(tuple(list(file_)[0].split(',')))
    target_ = st.selectbox("choose your target value",
                           tuple(list(dataframe)), index=0)

    st.title("Data Preprocessing")
    # 'outliers_threshold': 0.05,
    # 'normalize_method': 'zscore',
    # 'feature_selection_threshold': 0.8,
    # 'multicollinearity_threshold': 0.9,
    # 'pca_components': 0.99,
    v1 = st.checkbox("Remove Outliers", key='rem_outlier')
    if v1:
        v11 = st.slider("Outlier Threshold", min_value=0.0,
                        max_value=0.3, value=0.05, step=0.01)
    v2 = st.checkbox("Fix Imbalance", key='2')
    v3 = st.checkbox("Normalize", key='3')
    if v3:
        v33 = st.selectbox("Normalize Method", ("zscore",
                           "minmax", "maxabs", "robust",), index=0)
    v4 = st.checkbox("Feature Selection", key='4')
    if v4:
        v44 = st.slider("Feature Selection Threshold", min_value=0.0,
                        max_value=1.0, value=0.8, step=0.1)
    v5 = st.checkbox("Remove Muticolinearity", key='5')
    if v5:
        v55 = st.slider("Multicolinearity Threshold", min_value=0.0,
                        max_value=1.0, value=0.9, step=0.1)
    v6 = st.checkbox("PCA", key='6')
    if v6:
        v66 = st.slider("PCA components", min_value=0.0,
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
        best_model = compare_models()
        st.write(pd.DataFrame(models()))
        st.write("BEST MODEL")
        print("------------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-", best_model)
        st.write(best_model)
        # print(type(best_model))
        # save_model(best_model, 'my_best_pipeline')
