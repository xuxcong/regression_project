from model_test import ModelBase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = open("../Data/train.csv")
df = pd.read_csv(f)
f.close()
sexes = df["Sex"]
ages = df["Age"]
survivals = df["Survived"]
pclasses = df["Pclass"]

# Number of genders
n1 = 2
# Number of classes
n2 = 3
# Number of age partitions
n3 = 8
age_cuts = np.linspace(0,80,n3+1)
survival_ratio = np.zeros((n1,n2,n3))
for i in range(n1):
    for j in range(n2):
        for k in range(n3):
            if i==0:
                sex = "male"
            else:
                sex = "female"
            pclass = j+1
            low_age = age_cuts[k]
            high_age = age_cuts[k+1]
            idx = (ages>=low_age)&(ages<high_age)
            idx = idx & (sexes==sex)
            idx = idx & (pclasses==pclass)
            this_survivals = survivals[idx]
            survival_ratio[i,j,k]=np.mean(this_survivals)
        plt.subplot(2,3,i*3+j+1)
        plt.bar(age_cuts[0:n3],survival_ratio[i,j,:],align="edge",width=80/n3)
        plt.title("{0}:{1}".format(sex,str(pclass)))
        
plt.show()




