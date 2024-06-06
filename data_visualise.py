
import numpy as np
import csv

import matplotlib.pyplot as plt
from hmmlearn import hmm
np.random.seed(42)
import pandas as pd
import argparse

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--file', default="generated_data.csv", required=False, help='Data file')	
    parser.add_argument('--separate', default="no", required=False, help='Sep. plots?')	
    
    args = parser.parse_args()	
    return  args.file, args.separate
    
data_file, separate   =  ParseArguments()

data=pd.read_csv(data_file)

 

appliances = data.columns.values.tolist() 
 
appliances.remove('time')
## appliances.remove('site_meter')

n_samples=data.shape[0]

plt.figure()
## plt.plot(data['site_meter'], label="Site meter", alpha=0.3);


for appliance in appliances:
    if(separate=="yes"):
        plt.figure()
    plt.plot(data[appliance], label=appliance);
    if(separate=="yes"):
        plt.title(appliance)
        plt.legend()


if(separate=="no"):    
    plt.legend()

 
plt.show() 
