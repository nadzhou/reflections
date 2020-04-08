import pandas as pd 
import numpy as np
from pandas.plotting  import scatter_matrix



def correlations(filename): 
    data = pd.read_csv(filename)

    correlated = data.corr()

    scatter_matrix(data[data], figsize=(12, 8))
    # print(correlated['Variation score'].sort_values(ascending=False))


correlations("/home/nadzhou/Desktop/results.csv")