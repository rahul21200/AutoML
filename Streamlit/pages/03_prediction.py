import streamlit as st
import pandas as pd
from datetime import datetime

dataframe = pd.read_csv('temp.csv')
st.title('Predict your data')

option = st.radio('What is your ML model type?',
                  ('Classification', 'Regression'), index=0, horizontal=True)

if option == "Classification":
    from classification import DataPreProcessing
    from pycaret.classification import *
if option == "Regression":
    from regression import DataPreProcessing
    from pycaret.regression import *
st.header("Target")
target_ = st.selectbox("choose your target value",
                       tuple(list(dataframe)[::-1]), index=0)

st.title("Data Preprocessing")
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

submit = st.button("Submit", key="submit")
if submit:
    mod = DataPreProcessing(dataframe)
    if v1:
        mod.remove_outliers()
        mod.outliers_threshold(v11)
    if v2:
        mod.fix_imbalance()
    if v3:
        mod.normalize()
        mod.normalize_method(v33)
    if v4:
        mod.feature_selection()
        mod.feature_selection_threshold(v44)
    if v5:
        mod.remove_multicollinearity()
        mod.multicollinearity_threshold(v55)
    if v6:
        mod.pca()

    mod.setTarget(target_.strip())
    models_ = mod.allSetup()
    best_model = compare_models(verbose=False)
    model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_model(best_model, model_name)
    pull1 = pull()

    st.write(pull1)
    st.write("BEST MODEL")
    st.write(best_model)
    with open(model_name + ".pkl", "rb") as f:
        Confirm_download = st.download_button(
            "Download Model", f, file_name=f"{model_name}.pkl")
    if Confirm_download:
        os.remove(model_name + ".pkl")
