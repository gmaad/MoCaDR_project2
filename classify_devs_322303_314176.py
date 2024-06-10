import numpy as np
from hmmlearn import hmm
import pandas as pd
import random
import os
import argparse

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--train', default="house3_5devices_train.csv", required=False, help='Train data file')
    parser.add_argument('--test', default="test_folder", required=False, help='Test data file')
    parser.add_argument('--output', default="results.txt", required=False, help='Output file')

    args = parser.parse_args()
    return args.train, args.test, args.output

def prepare_data(data, index):
    return(np.array(data[data.columns[index]]))

def make_model(n_components, data):
    model = hmm.GaussianHMM(n_components=n_components,
                            random_state=random.randint(1, 1000), n_iter=1000)
    model.fit(data[:,None])
    return(model)

def evaluate_model(model, data):
    return(model.aic(data[:, None]))


### read in data ###
if __name__ == "__main__":

    train, test, output = ParseArguments()

    df  = pd.read_csv(train, delimiter=',')
    df = df.drop(columns=['time'])
    nrows, ncols = df.shape
    train_data = [prepare_data(df,i) for i in range(ncols)]

    # 1. fit the models to train data
    n_components = [11,12,11,10,9]
    devices = {0: "lighting2", 1: "lighting5", 2: "lighting4",
               3: "refrigerator", 4: "microwave"}
    models = [make_model(n_components[i], train_data[i]) for i in range(ncols)]

    # 2. assign model to the device

    folder_name = test
    path_to_folder = os.path.abspath(folder_name)
    files_in_folder = os.listdir(folder_name)
    path_to_files = [os.path.join(path_to_folder, x) for x in files_in_folder]

    # output file
    output_file = open(output, "w")
    output_file.write('file,dev_classified'+'\n')

    ind = 0
    for file in path_to_files:
        df2 = pd.read_csv(file, delimiter=',')
        df2 = df2.drop(columns=['time'])
        test_data = prepare_data(df2,0)

        scores = [evaluate_model(model, test_data) for model in models]
        device = devices[np.argmin(scores)]
        output_file.write(files_in_folder[ind] +','+device+'\n')
        ind = ind+1

    output_file.close()





