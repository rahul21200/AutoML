
# importing packages
import seaborn
import matplotlib.pyplot as plt
  
############# Main Section ############
# loading dataset using seaborn
df = seaborn.load_dataset('iris')
# pairplot with hue sex
seaborn.pairplot(df, hue ='species')
# to show
plt.show()