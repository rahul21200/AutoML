from pycaret.regression import *

import pandas as pd

# with open('my_best_pipeline.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data)
# x_test = pd.read_csv()
# print(data.predict([32, 3.5, 1.4, 0.2, "Iris-setosa"]))
# result = data.score(, Y_test)
lm = load_model("my_best_pipeline")
# plot_model(lm, plot='auc')
# print(lm)
# predictions = predict_model(lm, data=pd.read_csv("test.csv"))
# print(type(predictions))
# print(predictions.head())
data_unseen = pd.read_csv('test.csv')
unseen_predictions = predict_model(lm, data=data_unseen)
print(unseen_predictions.head())
