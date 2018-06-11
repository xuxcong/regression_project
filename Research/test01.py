
# Load model base
from model_test2 import ModelBase,summary
# Common model algorithms
from sklearn import svm,naive_bayes
import pandas as pd

from pdb import set_trace

data_x = ["IsAlone","Sex_Code","Pclass","Embarked_Code","Title_Code","FamilySize","AgeBin_Code","FareBin_Code"]

if __name__=="__main__":
    f = open("../Data/preprocessed_train.csv")
    train_df = pd.read_csv(f)
    f.close()
    f = open("../Data/preprocessed_test.csv")
    test_df = pd.read_csv(f)
    f.close()

    clf = svm.SVC()
    clf.fit(train_df[data_x],train_df["Survived"])
    results = clf.predict(test_df[data_x])
    output_dict = {}
    # set_trace()
    output_dict["PassengerId"] = test_df["PassengerId"]
    output_dict["Survived"] = pd.Series(results)
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv("../Data/predict.csv",index=False)