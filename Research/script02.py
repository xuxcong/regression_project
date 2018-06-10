from model_test import ModelBase,summary
import numpy as np
import pandas as pd
from pdb import set_trace

class GuessModel(ModelBase):
    """

    """
    def setup(self):
        # 2 sexes
        self.n1 = 2
        # 3 classes
        self.n2 = 3
        # number of age cuts
        self.n3 = 8
        #
        self.cuts = np.linspace(0,80,self.n3+1)
        # Parameters
        # The last slice of age dimension stores the probability of survival of those who did not have age record
        # We guess that those did not have age record most likely died in the catastraphe
        self.survival_prob = np.zeros((self.n1,self.n2,self.n3+1))

    def get_age_range(self,age):
        for i in range(self.n3):
            if age<=self.cuts[i+1]:
                return i
        # Don't have age record
        return self.n3

    def train(self,features,labels):
        # Retrieve features
        sexes = features["Sex"]
        ages = features["Age"]
        pclasses = features["Pclass"]
        # Parameters estimation
        for i in range(self.n1):
            if i==0:
                sex = "male"
            else:
                sex = "female"
            for j in range(self.n2):
                pclass = j+1
                for k in range(self.n3):
                    low_age = self.cuts[k]
                    high_age = self.cuts[k+1]
                    idx = (ages>=low_age)&(ages<high_age)
                    idx = idx & (sexes==sex)
                    idx = idx & (pclasses==pclass)
                    this_survivals = labels[idx]
                    # In case no record
                    if len(this_survivals)<=0:
                        self.survival_prob[i,j,self.n3]=0.5
                    else:
                        # Compute mean value, ignoring nan
                        self.survival_prob[i,j,k]=np.nanmean(this_survivals)

                # Consider those who did not have age record
                idx = np.isnan(ages)
                idx = idx & (sexes==sex)
                idx = idx & (pclasses==pclass)
                this_survivals = labels[idx]
                # In case no record
                if len(this_survivals)<=0:
                    self.survival_prob[i,j,self.n3]=0.5
                else:
                    self.survival_prob[i,j,self.n3]=np.nanmean(this_survivals)

    def predict(self,features):
        # Retrieve features
        sexes = features["Sex"]
        ages = features["Age"]
        pclasses = features["Pclass"]
        # Initialize labels
        n = len(features)
        labels = pd.Series([0]*n)
        # Generate random numbers
        rands = np.random.rand(n)

        for i in range(n):
            sex = sexes[i]
            age = ages[i]
            pclass = pclasses[i]-1
            if sex=="male":
                sex=0
            else:
                sex=1
            age = self.get_age_range(age)
            prob = self.survival_prob[sex,pclass,age]
            # set_trace()
            if rands[i]<prob:
                labels[i] = 1
        return labels

if __name__=="__main__":
    model = GuessModel()
    results = model.run()
    summary(results)