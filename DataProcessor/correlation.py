import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

names = ["NN","GLM","BernoulliNB","GaussianNB","LinearSVC","SVC","NuSVC",]

labels = pd.DataFrame()

for name in names:
    f = open("../Data/{0}_predict.csv".format(name))
    df = pd.read_csv(f)
    f.close()
    labels[name]=df["Survived"]

# Compute the correlation matrix of predictions from different models
cor = labels.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(cor,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
# Set up figure
# f,ax = plt.subplots(figsize=(11,9))
# Generate a custom diverging colormap
cmap = sb.diverging_palette(220,10,as_cmap=True)
# Plotting correlation heatmap
sb.heatmap(cor,mask=mask,cmap=cmap)
plt.show()
