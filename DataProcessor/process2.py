from basic_process import cross_validation
import json
import numpy as np

if __name__=="__main__":
    np.random.seed(960723)
    with open('../config.json', 'r') as f_config:
        config = json.load(f_config)
    print('configuration complete!')
    cross_validation('../Data/preprocessed_train.csv', '../Data/CrossValidation2', K=10)