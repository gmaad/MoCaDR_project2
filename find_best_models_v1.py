import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd
import random

#----------------------linear plots----------------------------------------
def find_best_model_for_train_data(data, model_name, max_components_nr):
    scores = list()
    models = list()
    aic = list()
    bic = list()

    for n_components in range(1, max_components_nr+1):
        model = hmm.GaussianHMM(n_components=n_components,
                                    random_state=random.randint(1,1000), n_iter=100)
        model.fit(data[:, None])
        models.append(model)
        scores.append(model.score(data[:, None]))
        aic.append(model.aic(data[:, None]))
        bic.append(model.bic(data[:, None]))

    best_model_score = [np.max(scores), np.argmax(scores)+1]
    best_model_aic = [np.min(aic),np.argmin(aic)+1]
    best_model_bic = [np.min(bic),np.argmin(bic)+1]
    res = {"score": best_model_score,
           "aic": best_model_aic,
           "bic": best_model_bic}

    results = pd.DataFrame(res, index = ["statistics_value", "n_components"])

    plots(aic, bic, scores, model_name, max_components_nr)
    return(results)

def plots(aic, bic, scores, model_name, max_components_nr):
    xs = [i for i in range(1, max_components_nr+1)]
    plt.plot(xs,aic, color = 'blue', label= "AIC", marker = "o")
    plt.plot(xs,bic, color="orange", linestyle = 'dashed', label= "BIC", marker = "o")
    plt.plot(xs,scores, color = 'red', label = "LL", marker = "o")
    plt.xlabel("Number of components")
    plt.ylabel("Statistics value")
    plt.title("Model's information statistics for "+ model_name)
    plt.legend()

    plt.show()


###---------------boxplots------------------------
def find_best_model_for_train_data_v2(data,model_name, max_components_nr=30, max_random_states_nr=10):
    scores = np.zeros([max_components_nr, max_random_states_nr])
    aic = np.zeros([max_components_nr, max_random_states_nr])
    bic = np.zeros([max_components_nr, max_random_states_nr])

    for n_components in range(1, max_components_nr):
        for idx in range(max_random_states_nr):
            model = hmm.GaussianHMM(n_components=max_components_nr,
                                    random_state=random.randint(1,1000), n_iter=100)
            model.fit(data[:, None])
            scores[n_components, idx] = model.score(data[:, None])
            aic[n_components, idx] = model.aic(data[:, None])
            bic[n_components, idx] = model.bic(data[:, None])

    boxplots(aic, bic, scores, model_name)

def boxplots(aic, bic, scores, model_name):
    nrow,ncol = aic.shape
    plt.boxplot(aic.transpose())
    plt.xlabel("Number of components")
    plt.ylabel("Statistics value")
    plt.title("AIC for " + model_name)
    plt.show()

    plt.boxplot(bic.transpose())
    plt.xlabel("Number of components")
    plt.ylabel("Statistics value")
    plt.title("BIC for " + model_name)
    plt.show()

    plt.boxplot(scores.transpose())
    plt.xlabel("Number of components")
    plt.ylabel("Statistics value")
    plt.title("LL for " + model_name)
    plt.show()


###-----------------------------main----------------------------------
df = pd.read_csv('house3_5devices_train.csv', delimiter=',')
nrows,ncols = df.shape #col 0  == time - not used
indexes_cols = list(range(1, ncols))

devices = {1:"lighting2", 2:"lighting5", 3:"lighting4",
           4:"refrigerator", 5:"microwave"}


k = 1
find_best_model_for_train_data_v2(np.array(df[df.columns[k]]), devices[k])


'''
 LIGHTING2
                         score            aic          bic
statistics_value  56477.121548 -112310.243097 -109857.9752
n_components         17.000000      17.000000      17.0000


 LIGHTING5
                         score            aic            bic
statistics_value  67655.215541 -133746.431081 -128503.119425
n_components         27.000000      27.000000      21.000000


LIGHTING4
                         score            aic           bic
statistics_value  51814.388763 -101950.777527 -96100.106334
n_components         28.000000      28.000000     24.000000


REFRIGERATOR
                         score           aic           bic
statistics_value  34633.888012 -67349.776023 -61344.258202
n_components         30.000000     30.000000     13.000000


MICROWAVE
                        score            aic           bic
statistics_value  81773.15986 -162668.319719 -159625.50441
n_components         20.00000      20.000000      13.00000
'''

