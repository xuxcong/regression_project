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

import pandas as pd

# Configure Visualization Defaults
mpl.style.use("ggplot")
sb.set_style("white")

# Load model base
from model_test import ModelBase



class SVCModel(ModelBase):
    def setup(self):
        pass
    
    def train(self,features,labels):
        pass

    def predict(self,features):
        
        pass

class NuSVCModel(ModelBase):
    def setup(self):
        pass
    
    def train(self,features,labels):
        pass

    def predict(self,features):
        pass

class LinearSVCModel(ModelBase):
    def setup(self):
        pass
    
    def train(self,features,labels):
        pass

    def predict(self,features):
        pass

class BernoulliNBModel(ModelBase):
    def setup(self):
        pass
    
    def train(self,features,labels):
        pass

    def predict(self,features):
        pass

class GaussianNBModel(ModelBase):
    def setup(self):
        pass
    
    def train(self,features,labels):
        pass

    def predict(self,features):
        pass