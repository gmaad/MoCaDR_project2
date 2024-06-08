import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
import random
import os

def prepare_data(data, index):
    return(np.array(data[data.columns[index]]))

def make_model(n_components, data):
    model = hmm.GaussianHMM(n_components=n_components,
                            random_state=random.randint(1, 1000), n_iter=1000)
    model.fit(data[:,None])
    return(model)

def statistics(model, data):
    return([model.score(data[:, None]),
            model.aic(data[:, None]),
            model.bic(data[:, None])])

def evaluate_model(model, data):
    return(model.bic(data[:, None]))


### read in data ###
if __name__ == "__main__":
    df  = pd.read_csv('house3_5devices_train.csv', delimiter=',')
    df = df.drop(columns=['time'])
    nrows, ncols = df.shape
    train_data = [prepare_data(df,i) for i in range(ncols)]


    # 1. fit the models to train data
    n_components = [11,3,11,5,4]
    devices = {0: "lighting2", 1: "lighting5", 2: "lighting4",
               3: "refrigerator", 4: "microwave"}
    models = [make_model(n_components[i], train_data[i]) for i in range(ncols)]

    # 2. assign model to the device

    folder_name = 'test_folder2'
    path_to_folder = os.path.abspath(folder_name)
    files_in_folder = os.listdir(folder_name)
    path_to_files = [os.path.join(path_to_folder, x) for x in files_in_folder]

    list_of_df = list()
    for file in path_to_files:
        df2 = pd.read_csv(file, delimiter=',')
        df2 = df2.drop(columns=['time'])
        test_data = prepare_data(df2,0)

        scores = [evaluate_model(model, test_data) for model in models]
        device = devices[np.argmin(scores)]

        print(device)


