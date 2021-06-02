import numpy as np
import pandas as pd

data=pd.read_csv('train.csv')
data=data[data.isnull().sum(axis=1) <2]
data=data.iloc[:,2:]
data=data.dropna()




def correlation(dataset,threshold):
        col_corr=set()
        corr_matrix=dataset.corr()
        for i in range(len(corr_matrix.columns)):
                for j in range(i):
                        if abs(corr_matrix.iloc[i,j])> threshold:
                                colname=corr_matrix.columns[i]
                                col_corr.add(colname)
        return col_corr
corr_features=correlation(data,0.7)
print(corr_features)
#data=data.drop(corr_features, axis=1)

#print(corr_features)

"""

from sklearn.feature_selection import SelectKBest, mutual_info_regression
selector = SelectKBest(mutual_info_regression, k=4)
selector.fit(data,ydata)
data.columns[selector.get_support()]







"""
