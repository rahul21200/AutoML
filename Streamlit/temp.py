
# importing packages

import seaborn
import pandas as pd
import matplotlib.pyplot as plt
from AutoClean import AutoClean
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()

df = AV.AutoViz('iris.csv',verbose=2,chart_format='png')

############# Main Section ############
# loading dataset using seaborn
# df = seaborn.load_dataset('iris')
# df = pd.read_csv('train-clean.csv')
# pipeline = AutoClean(df,encode_categ=[False])
# print(pipeline.output)
# # pairplot with hue sex
# seaborn.pairplot(pipeline.output, hue ='Sex')
# # to show
# plt.show(block=True)
# plt.pause(10)
