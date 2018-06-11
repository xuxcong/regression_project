# Common model algorithms
from sklearn import svm,naive_bayes

# Common model helpers
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import feature_selection,model_selection,metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sb
import numpy as np
import pandas as pd

# Configure Visualization Defaults
mpl.style.use("ggplot")
sb.set_style("white")

# Load model base
from model_test2 import ModelBase,summary
from pdb import set_trace

data_x = ["IsAlone","Sex_Code","Pclass","Embarked_Code","Title_Code","FamilySize","AgeBin_Code","FareBin_Code"]

class SVCModel(ModelBase):
    def setup(self):
        self.clf = svm.SVC()
    
    def train(self,features,labels):
        self.clf.fit(features[data_x],labels)

    def predict(self,features):
        return pd.Series(self.clf.predict(features[data_x]))

class NuSVCModel(ModelBase):
    def setup(self):
        self.clf = svm.NuSVC()
    
    def train(self,features,labels):
        self.clf.fit(features[data_x],labels)

    def predict(self,features):
        return pd.Series(self.clf.predict(features[data_x]))

class LinearSVCModel(ModelBase):
    def setup(self):
        self.clf = svm.LinearSVC()

    def train(self,features,labels):
        self.clf.fit(features[data_x],labels)

    def predict(self,features):
        return pd.Series(self.clf.predict(features[data_x]))

class BernoulliNBModel(ModelBase):
    def setup(self):
        self.clf = naive_bayes.BernoulliNB()
    
    def train(self,features,labels):
        self.clf.fit(features[data_x],labels)

    def predict(self,features):
        return pd.Series(self.clf.predict(features[data_x]))

class GaussianNBModel(ModelBase):
    def setup(self):
        self.clf = naive_bayes.GaussianNB()
    
    def train(self,features,labels):
        self.clf.fit(features[data_x],labels)

    def predict(self,features):
        return pd.Series(self.clf.predict(features[data_x]))

if __name__=="__main__":
    # Create models
    svc_mdl = SVCModel()
    nusvc_mdl = NuSVCModel()
    linearsvc_mdl = LinearSVCModel()
    bernoullinb_mdl = BernoulliNBModel()
    gaussian_mdl = GaussianNBModel()

    # Ensemble models
    mdls = {"SVC":svc_mdl,"NuSVC":nusvc_mdl,"LinearSVC":linearsvc_mdl,
    "BernoulliNB":bernoullinb_mdl,"GaussianNB":gaussian_mdl}

    names = mdls.keys()
    # Perform cross validation on these models
    results = {}
    correct_rates = {}
    for k in names:
        results[k]=mdls[k].run_cv()
        print(k)
        correct_rates[k]=summary(results[k])[0]
    
    # Plotting correct rates comparison
    name_list = [v for v in names]
    plt.bar(range(1,len(names)+1),[correct_rates[name]*100 for name in name_list],width = 0.8,tick_label=name_list)
    plt.ylabel("Correct Rate [%]")
    plt.show()

    # Perform main testing with these models
    guesses = pd.DataFrame()
    for k in names:
        guesses[k]=mdls[k].run_main("{0}_predict".format(k))
    # Compute the correlation matrix of predictions from different models
    cor = guesses.corr()
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
