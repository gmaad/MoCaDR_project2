import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score

covariance_type_options = ['spherical', 'diag', 'full', 'tied']

def prepare_data(data, index):
    return(np.array(data[data.columns[index]]))

def evaluate_model(X, n_components, covariance_type='full'):
    hmm_model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=100)
    hmm_model.fit(X[:,None])
    bic = hmm_model.bic(X[:,None])
    return bic

def cross_validation(data, n_components_range):
    kf = KFold(n_splits=5)
    bics = list()
    parameters = list()

    for n_components in n_components_range:
        for cov_type in covariance_type_options:
            scores = list()
            for train_index, test_index in kf.split(data):
                X_train= data[train_index]
                score = evaluate_model(X_train, n_components, cov_type)
                scores.append(score)
            avg_bic = np.mean(scores)
            bics.append(avg_bic)
            parameters.append((n_components, cov_type))

    index = np.argmax(bics)
    res = parameters[index]
    return(res)


if __name__ == "__main__":

    df  = pd.read_csv('house3_5devices_train.csv', delimiter=',')
    df = df.drop(columns=['time'])
    data = prepare_data(df, 0)

    print(cross_validation(data, [4,11]))
    print(evaluate_model(data,4))
    print(evaluate_model(data, 11))




