import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

def preprocess(features):
    new_feas = features.copy()
    # Clean data
    # complete missing age value with median
    new_feas["Age"].fillna(new_feas["Age"].median(),inplace=True)
    # complete embarked with mode
    new_feas["Embarked"].fillna(new_feas["Embarked"].mode()[0],inplace=True)
    # complete missing fare with median
    new_feas["Fare"].fillna(new_feas["Fare"].median(),inplace=True)
    # Drop unused columns
    drop_columns = ["PassengerId","Cabin","Ticket"]
    new_feas.drop(drop_columns,axis=1,inplace=True)

    # Feature engineer
    # Compute the size of family of each individual
    new_feas["FamilySize"] = new_feas["SibSp"]+new_feas["Parch"]+1
    new_feas["IsAlone"] = 1
    new_feas["IsAlone"].loc[new_feas["FamilySize"]>1]=0
    new_feas["Title"] = new_feas["Name"].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
    # Using quantiles to cut Fare into equal size bins
    # indicating different level of fares
    new_feas["FareBin"] = pd.qcut(new_feas["Fare"],4)
    # linearly cut age into ranges
    # indicating different age stage
    new_feas["AgeBin"] = pd.cut(new_feas["Age"].astype(int),5)

    # Clean up rare title names
    # Statistically minimum
    stat_min = 10
    # For title names that less than 10 people have it
    # we will deem it as rare and categorize it into Misc titles
    title_names = (new_feas["Title"].value_counts()<stat_min)
    new_feas["Title"]=new_feas["Title"].apply(lambda x:"Misc" if title_names.loc[x]==True else x)

    label = LabelEncoder()
    new_feas["Sex_Code"] = label.fit_transform(new_feas["Sex"])
    new_feas["Embarked_Code"]=label.fit_transform(new_feas["Embarked"])
    new_feas["Title_Code"]=label.fit_transform(new_feas["Title"])
    new_feas["AgeBin_Code"]=label.fit_transform(new_feas["AgeBin"])
    new_feas["FareBin_Code"]=label.fit_transform(new_feas["FareBin"])

    return new_feas

if __name__=="__main__":
    f = open("../Data/train.csv")
    train_df = pd.read_csv(f)
    f.close()
    # Preprocess
    new_train_df = preprocess(train_df)
    new_train_df.to_csv("../Data/preprocessed_train.csv")