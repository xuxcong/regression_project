from collections import defaultdict
import pandas as pd
import os
import numpy as np
import json
import codecs


def cross_validation(input_file, output_folder, K=10):
    """
    Divide the training file for cross validation.
    Keep the ratio of each label in each fold the same as in the input file.
    """


    df = pd.read_csv(input_file) # type: pandas.core.frame.DataFrame
    label2idx = defaultdict(list)  # type: dict[str,list[int]]
    dev_range = defaultdict(list)  # type: dict[str,list[tuple[int,int]]]

    for index, row in df.iterrows():
        idx, label = row[0], row[1]
        label2idx[label].append(int(idx) - 1)


    # specify the starting and ending indices of dev-set in each fold for each label
    for label in label2idx:
        np.random.shuffle(label2idx[label])
        N = len(label2idx[label])
        dev_len = int(N / K)
        for i in range(K):
            start = i * dev_len
            end = (i + 1) * dev_len if i < K - 1 else N
            dev_range[label].append((start, end))

    for i in range(K):
        cur_folder = os.path.join(output_folder, 'fold_{:02d}'.format(i + 1))
        if not os.path.exists(cur_folder):
            os.makedirs(cur_folder)

        train_data = pd.DataFrame(columns = list(df.iloc[0].index))
        dev_data = pd.DataFrame(columns = list(df.iloc[0].index))

        for label in label2idx:
            for j, (a, b) in enumerate(dev_range[label]):
                for idx in label2idx[label][a:b]:
                    if i == j:
                        # this is for dev in current fold
                        dev_data = dev_data.append(df.iloc[idx], ignore_index = True)
                    else:
                        train_data = train_data.append(df.iloc[idx], ignore_index = True)
                
        dev_data.to_csv(os.path.join(cur_folder, 'dev.csv'), index=False)
        train_data.to_csv(os.path.join(cur_folder, 'train.csv'), index=False)




if __name__ == '__main__':
    np.random.seed(960723)
    with open('../config.json', 'r') as f_config:
        config = json.load(f_config)
    print('configuration complete!')
    cross_validation('../Data/train.csv', '../Data/CrossValidation', K=10)
    
